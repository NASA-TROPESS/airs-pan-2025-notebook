import itertools as it
from math import prod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree

from ..features import FeatureSet
from .metrics import PerformanceMetric, TruthRateMetric, TrainingFlag

def train_decision_tree(training_input_df: pd.DataFrame, is_good: pd.Series, **tree_kws) -> tree.DecisionTreeClassifier:
    """Train a decision tree to predict whether an AIRS XPAN800 value is good quality.

    Parameters
    ----------
    training_input_df
        A dataframe containing ONLY the input variables for the model.

    is_good
        A series that has the actual quality flags corresponding to the
        soundings in ``training_input_df``.

    tree_kws
        Extra keywords to pass to :class:`tree.DecisionTreeClassifier`.
        This function uses the default parameters for that class if none
        are given here.

    Returns
    -------
    classifier
        The trained model.
    """
    clf = tree.DecisionTreeClassifier(**tree_kws)
    clf = clf.fit(training_input_df.to_numpy(), is_good.to_numpy())
    return clf


def make_model(training_df_full: pd.DataFrame, input_features: FeatureSet, flagger: TrainingFlag,
               tree_kws = dict()):
    training_df_full = training_df_full.dropna()
    training_df = input_features.subset_df(training_df_full)
    is_good = flagger.make_training_flag(training_df_full)
    clf = train_decision_tree(training_df, is_good, **tree_kws)
    
    return clf


def model_prediction(model: tree.DecisionTreeClassifier, val_df_full: pd.DataFrame, features: FeatureSet):
    val_df_full = val_df_full.dropna()
    val_df = features.subset_df(val_df_full)
    return model.predict(val_df.to_numpy())


def train_tree_and_plot_val(
    training_df_full: pd.DataFrame, val_df_full: pd.DataFrame, input_features: FeatureSet,
    flagger: TrainingFlag, axs=None, **tree_kws
):
    """A convenience function that both trains the decision tree and does simple validation on it.

    Parameters
    ----------
    training_df_full
        A dataframe that contains both the input features the model will use and the AIRS and CrIS
        XPAN800 values used to compute the true quality flag. This input will be used to train the model.

    val_df_full
        A dataframe like ``training_df_full``, but with data to be used to validate the model. There should
        be no overlap in soundings between this and ``training_df_full``.

    input_features
        A :class:`FeatureSet` instance that will extract the model input features from ``training_df_full``
        and ``val_df_full``.

    flagger
        A :class:`TrainingFlag` instance that will compute the true quality flags from the input dataframes.

    axs
        A sequence of two axes to plot the validation results in; the first will have a scatter plot of the
        training data (AIRS XPAN vs. CrIS XPAN) colored by whether it is correctly or incorrectly good or bad,
        the second will similarly show the results with the validation data. If not given, they will be created.

    tree_kws
        Additional keywords for the :class:`tree.DecisionTreeClassifier`, passed when it is initialized.

    Returns
    -------
    classifier
        The trained model.
    """
    training_df_full = training_df_full.dropna()
    val_df_full = val_df_full.dropna()
    
    training_df = input_features.subset_df(training_df_full)
    val_df = input_features.subset_df(val_df_full)
    
    is_good = flagger.make_training_flag(training_df_full)
    clf = train_decision_tree(training_df, is_good, **tree_kws)
    
    is_good_val = flagger.make_training_flag(val_df_full).to_numpy()
    is_good_pred = clf.predict(training_df.to_numpy())
    is_good_val_pred = clf.predict(val_df.to_numpy())
    
    
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(12,4))
    
    metric = TruthRateMetric()
    training_metric_arrays = metric.index_arrays(is_good, is_good_pred)
    val_metric_arrays = metric.index_arrays(is_good_val, is_good_val_pred)
    
    ax = axs[0]
    airs_xpan, cris_xpan = flagger.get_xpan_columns(training_df_full)
    _plot_tree_success(airs_xpan, cris_xpan, training_metric_arrays, ax=ax)
    ax.set_title('Training data')

    
    ax = axs[1]
    airs_xpan, cris_xpan = flagger.get_xpan_columns(val_df_full)
    _plot_tree_success(airs_xpan, cris_xpan, val_metric_arrays, ax=ax)
    ax.set_title('Validation data')
    
    plt.suptitle(str(flagger))
    
    return clf


def _plot_tree_success(airs_xpan: pd.Series, cris_xpan: pd.Series, truth_met_arrs: dict, ax=None):
    """Plot the correct and incorrect good and bad flags as a scatter plot of AIRS vs. CrIS XPAN.

    Parameters
    ----------
    airs_xpan
        The AIRS XPAN800 values.

    cris_xpan
        The CrIS XPAN800 values (interpolated to the AIRS sounding locations).

    truth_met_arrs
        The dictionary of true good, true bad, false good, and false bad logical index arrays
        returned by :meth:`TruthRateMetric.index_arrays`.

    ax
        Optional, if given, axes to plot into.
    """
    ax = ax or plt.gca()
    n_tot = sum(v.sum() for v in truth_met_arrs.values())
    assert n_tot == list(truth_met_arrs.values())[0].size, 'Some elements uncategorized or double categoriezed'
    
    for k, inds in truth_met_arrs.items():
        label = k.capitalize().replace('_', ' ')
        f = inds.sum() / n_tot * 100
        x = cris_xpan.loc[inds]
        y = airs_xpan.loc[inds]
        ax.plot(x, y, marker='.', markersize=2, ls='none', label=f'{label} ({f:.1f}%)')
        
    ax.set_xlabel('CrIS interpolated XPAN800')
    ax.set_ylabel('AIRS XPAN800')
    ax.legend()


def optimize_tree(training_df, val_df, tree_params: dict, input_features: FeatureSet, flagger: TrainingFlag,
                  metric: PerformanceMetric = TruthRateMetric()):
    pkeys = list(tree_params.keys())
    pvals = [tree_params[k] for k in pkeys]
    n = prod(len(v) for v in pvals)
    df = pd.DataFrame(columns=pkeys + ['nodes', 'metric'] + list(metric.extra_metric_names), index=range(n))
    
    for i, opts in enumerate(it.product(*pvals)):
        kws = {k: v for k, v in zip(pkeys, opts)}
        model = make_model(training_df, input_features, flagger, tree_kws=kws)
        extra_values = metric.calc_extra_metrics(
            model=model,
            val_df_full=val_df,
            features=input_features,
            flagger=flagger
        )
        for k, v in kws.items():
            df.loc[i,k] = v
        df.loc[i,'nodes'] = model.tree_.node_count
        df.loc[i,'metric'] = metric.validate(
            model=model,
            val_df_full=val_df,
            features=input_features,
            flagger=flagger
        )
        for k, v in extra_values.items():
            df.loc[i,k] = v
    return df


def plot_optimization_2d(opt_df, x_col, y_col, z_col='metric', na_fill=999, clabel=None, ax=None, **pcolor_kws):
    opt_df = opt_df.fillna(na_fill)
    xvals = np.sort(opt_df[x_col].unique())
    yvals = np.sort(opt_df[y_col].unique())
    zvals = np.full([xvals.size, yvals.size], np.nan)
    
    for (x, y), sub_df in opt_df.groupby([x_col, y_col]):
        i = np.flatnonzero(xvals == x).item()
        j = np.flatnonzero(yvals == y).item()
        zvals[i,j] = sub_df.mean()[z_col]
        
    xvals, xlabels = _replace_na_fill_with_max(xvals, na_fill)
    yvals, ylabels = _replace_na_fill_with_max(yvals, na_fill)
        
    ax = ax or plt.gca()
    h = ax.pcolormesh(xvals, yvals, zvals.T, **pcolor_kws)
    plt.colorbar(h, ax=ax, label=clabel or z_col)
    ax.set_xlabel(x_col)
    ax.set_xticks(xvals)
    if xlabels:
        ax.set_xticklabels(xlabels)
    ax.set_ylabel(y_col)
    ax.set_yticks(yvals)
    if ylabels:
        ax.set_yticklabels(ylabels)


def _replace_na_fill_with_max(vals, na_fill):
    i = np.flatnonzero(vals == na_fill)
    if i.size == 0:
        return vals, None
    
    i = i.item()
    m = np.max(vals[vals != na_fill])
    d = np.max(np.diff(np.sort(vals[vals != na_fill])))
    vals[i] = m + d
    labels = [str(v) for v in vals]
    labels[i] = 'Unlim.'
    return vals, labels


def plot_optimization_6panel(optimization_df, metric_vmin=None, metric_vmax=None):
    _, axs = plt.subplots(3, 2, figsize=(12,12))
    
    plot_optimization_2d(optimization_df, 'max_depth', 'max_leaf_nodes', clabel=TruthRateMetric().label, ax=axs[0,0], vmin=metric_vmin, vmax=metric_vmax)
    plot_optimization_2d(optimization_df, 'max_depth', 'max_leaf_nodes', z_col='nodes', clabel='# nodes', ax=axs[0,1], vmax=400, cmap='Purples')
    plot_optimization_2d(optimization_df, 'max_depth', 'max_leaf_nodes', z_col='true_good', ax=axs[1,0], vmin=0, vmax=1, cmap='Greens')
    plot_optimization_2d(optimization_df, 'max_depth', 'max_leaf_nodes', z_col='true_bad', ax=axs[1,1], vmin=0, vmax=1, cmap='Greens')
    plot_optimization_2d(optimization_df, 'max_depth', 'max_leaf_nodes', z_col='false_good', ax=axs[2,0], vmin=0, vmax=1, cmap='Reds')
    plot_optimization_2d(optimization_df, 'max_depth', 'max_leaf_nodes', z_col='false_bad', ax=axs[2,1], vmin=0, vmax=1, cmap='Reds')
