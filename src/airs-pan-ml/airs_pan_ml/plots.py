import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import FeatureSet
from .binary_flag.metrics import TrainingFlag


def plot_filtering_val_maps(clf, val_df_full, input_vars: FeatureSet, flagger: TrainingFlag, state_borders: bool=False, country_borders: bool = False, axs=None, **scatter_kw):
    val_df_full, is_good_pred, is_good_true, xx_all = _filtering_setup_helper(
        clf=clf,
        val_df_full=val_df_full,
        input_vars=input_vars,
        flagger=flagger
    )
    if axs is None:
        _, axs = plt.subplots(2, 2, figsize=(16,10), subplot_kw={'projection': ccrs.PlateCarree()})
        axs = axs.flatten()
    else:
        axs = axs.flatten()
        
    ax = axs[0]
    _map_helper(val_df_full, '_airs', xx_all, ax, state_borders, country_borders, **scatter_kw)
    ax.set_title('AIRS XPAN: no filter')
    
    ax = axs[1]
    if ax is not None:
        # Special case, this one can be ignored when plotting with `plot_filtering_val_maps_with_orig_cris`
        _map_helper(val_df_full, '_cris', xx_all, ax, state_borders, country_borders, **scatter_kw)
        ax.set_title('CrIS XPAN')
    
    ax = axs[2]
    _map_helper(val_df_full, '_airs', is_good_true, ax, state_borders, country_borders, **scatter_kw)
    ax.set_title(f'AIRS XPAN - true {flagger} filter')

    ax = axs[3]
    _map_helper(val_df_full, '_airs', is_good_pred, ax, state_borders, country_borders, **scatter_kw)
    ax.set_title(f'AIRS XPAN - predicted {flagger} filter')


def plot_filtering_val_maps_with_orig_cris(clf, val_df_full, orig_cris_df, input_vars: FeatureSet, flagger: TrainingFlag, include_interp_cris: bool = True,
                                           state_borders: bool = False, country_borders: bool = False, **scatter_kw):
    if include_interp_cris:
        _, axs = plt.subplots(2, 3, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        axs_subset = np.array([axs[0,0], axs[0,1], axs[1,0], axs[1,1]])
        cris_ax = axs[0,2]
    else:
        _, axs = plt.subplots(2, 2, figsize=(13.33, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        axs_subset = np.array([axs[0,0], None, axs[1,0], axs[1,1]])
        cris_ax = axs[0,1]
        
    plot_filtering_val_maps(clf, val_df_full, input_vars, flagger, axs=axs_subset, state_borders=state_borders, country_borders=country_borders, **scatter_kw)

    if 'qualflag' in orig_cris_df.columns:
        print('Filtering original cris data to qualflag == 1')
        xx = orig_cris_df['qualflag'] == 1
    else:
        print('Not filtering original CrIS data on quality')
        xx = np.ones(orig_cris_df.shape[0], dtype=bool)

    _map_helper(orig_cris_df, '', xx, cris_ax, state_borders, country_borders, **scatter_kw)
    cris_ax.set_title('CrIS XPAN (not interpolated)')

    if include_interp_cris:
        axs[1,2].axis('off')


def plot_multiple_model_filtering_maps(models_and_features: dict, val_df_full: pd.DataFrame, flagger: TrainingFlag, true_col_idx: int = -1,
                                       plot_true: bool = True, state_borders: bool = False, country_borders: bool = False, **scatter_kw):
    nmod = len(models_and_features)
    if true_col_idx < 0:
        # Needed for the equality test in the model loop
        true_col_idx = nmod + true_col_idx
    if plot_true:
        _, axs = plt.subplots(2, nmod, figsize=(6.7*nmod, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        pred_axs = axs[0,:]
        true_axs = axs[1,:]
    else:
        _, axs = plt.subplots(1, nmod, figsize=(6.7*nmod, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        pred_axs = axs
        true_axs = None

    for imod, (title, m_and_f) in enumerate(models_and_features.items()):
        this_vdf_full, this_good_pred, this_good_true, _ = _filtering_setup_helper(
            clf=m_and_f['clf'],
            val_df_full=val_df_full,
            input_vars=m_and_f['features'],
            flagger=flagger
        )

        ax = pred_axs[imod]
        _map_helper(this_vdf_full, '_airs', this_good_pred, ax=ax, state_borders=state_borders, country_borders=country_borders, **scatter_kw)
        ax.set_title(f'{title} - predicted')

        if plot_true and imod == true_col_idx:
            ax = true_axs[imod]
            _map_helper(this_vdf_full, '_airs', this_good_true, ax=ax, state_borders=state_borders, country_borders=country_borders, **scatter_kw)
            ax.set_title('True filtering')
        elif plot_true:
            true_axs[imod].axis('off')


def plot_feature_importance(rfe_df, ax=None, **style):
    p = np.flip(rfe_df['performance'].to_numpy())
    xt = np.flip(rfe_df.index)
    x = np.arange(xt.size)
    
    style.setdefault('marker', 'o')
    ax = ax or plt.gca()
    ax.plot(xt, p*100, **style)
    ax.set_xticks(x)
    ax.set_xticklabels(xt)
    ax.set_ylabel('Truth rate (%)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.grid()            
        
    

def _map_helper(df, suffix, xx, ax, state_borders, country_borders, **scatter_kw):
    h = ax.scatter(
        df.loc[xx, f'lon{suffix}'],
        df.loc[xx, f'lat{suffix}'],
        c=df.loc[xx, f'xpan{suffix}'],
        s=2, **scatter_kw
    )
    plt.colorbar(h, ax=ax, label='XPAN800')
    ax.coastlines()
    if state_borders:
        ax.add_feature(cfeat.STATES)
    if country_borders:
        ax.add_feature(cfeat.BORDERS)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False


def _filtering_setup_helper(clf, val_df_full, input_vars, flagger):
    val_df_full = val_df_full.dropna()
    val_df = input_vars.subset_df(val_df_full)
    is_good_pred = clf.predict(val_df.to_numpy())
    is_good_true = flagger.make_training_flag(val_df_full)
    xx_all = np.ones(is_good_pred.size, dtype=bool)
    return val_df_full, is_good_pred, is_good_true, xx_all
