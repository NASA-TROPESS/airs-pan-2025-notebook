import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..features import FeatureSet

from typing import Optional, Sequence, Tuple

class TrainingFlag:
    """A class that computes a binary flag for good or bad quality
    based on the difference between AIRS and CrIS XPAN.

    Parameters
    ----------
    max_abs_diff
        The maximum difference in XPAN800 allowed between the AIRS and CrIS
        values for an AIRS sounding to be considered good. The difference will
        always be positive. If left as ``None``, this limit will not be applied.
        In most cases, the unit will be ppb, but it will depend on the files
        from which data was loaded.
    
    max_rel_diff
        The maximum relative difference between AIRS and CrIS XPAN800 values
        for and AIRS sounding to be considered good. This is fractional, not
        percentage, so a value of 1 here means 100% different. The value will
        be |AIRS - CrIS| / |CrIS|. If left as ``None``, this limit will not be
        applied.

    op
        When both ``max_abs_diff`` and ``max_rel_diff`` given, this controls how
        they are combined. The default "and" means a sounding is good only if both
        the absolute and relative differences are below their respective thresholds.
        If this is "or", then a sounding is good if either difference is below its
        threshold.

    airs_pan_colname
        The name of the column in the dataframe to be given to ``make_training_flag``
        that contains the AIRS XPAN800.

    cris_pan_colname
        The name of the column in the dataframe to be given to ``make_training_flag``
        that contains the CrIS XPAN800, interpolated to the AIRS sounding locations.
    """
    def __init__(
        self,
        max_abs_diff: Optional[float] = None,
        max_rel_diff: Optional[float] = None,
        op: str = 'and',
        airs_pan_colname: str = 'xpan_airs',
        cris_pan_colname: str = 'xpan_cris_interp',
    ):
        if op not in {'and', 'or'}:
            raise ValueError('op must be one of "and", "or"')
        if max_abs_diff is None and max_rel_diff is None:
            raise ValueError('One of max_abs_diff and max_rel_diff must be given')
            
        self.op_and = op == 'and'
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.airs_pan_colname = airs_pan_colname
        self.cris_pan_colname = cris_pan_colname
        
    def __str__(self):
        if self.max_rel_diff is None:
            return f'Abs. diff $\\leq$ {self.max_abs_diff}'
        elif self.max_abs_diff is None:
            return f'Rel. diff $\\leq$ {self.max_rel_diff}'
        else:
            op = '&' if self.op_and else '|'
            return f'Abs. $\\leq$ {self.max_abs_diff} {op} rel. $\\leq$ {self.max_rel_diff}'

    def get_xpan_columns(self, feature_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Return the AIRS and CrIS XPAN series from ``feature_df``.
        """
        return feature_df[self.airs_pan_colname], feature_df[self.cris_pan_colname]
        
    def make_training_flag(self, feature_df: pd.DataFrame) -> pd.Series:
        """Returns a series that will be ``True`` for good AIRS soundings and ``False``
        for bad soundings.

        Parameters
        ----------
        feature_df
            A dataframe containing at least the AIRS and CrIS XPAN800 values.

        Returns
        -------
        is_good
            A series indicating if each of the soundings in ``feature_df`` are good quality.
        """
        diff = (feature_df[self.airs_pan_colname] - feature_df[self.cris_pan_colname]).abs()
        if self.max_abs_diff is not None:
            ok_abs = diff <= self.max_abs_diff
        else:
            ok_abs = pd.Series(np.ones(diff.size, dtype=bool), index=feature_df.index)
            
        if self.max_rel_diff is not None:
            reldiff = diff / feature_df[self.cris_pan_colname].abs()
            ok_rel = reldiff <= self.max_rel_diff
        else:
            ok_rel = pd.Series(np.ones(diff.size, dtype=bool), index=feature_df.index)
            
        if self.op_and:
            return ok_abs & ok_rel
        else:
            return ok_abs | ok_rel
        

class PerformanceMetric:
    """A base class for metrics to evaluate different model performances by.
    """
    
    def validate(self, model: DecisionTreeClassifier, val_df_full: pd.DataFrame, features: FeatureSet, flagger: TrainingFlag):
        """Given a model, inputs, and true flags, calculate the performance metric for the model.

        Parameters
        ----------
        model
            A trained model.

        val_df_full
            A dataframe containing both the AIRS & CrIS XPAN800 values and the input features
            used by the model.

        features
            A :class:`FeatureSet` instance that extracts the input features for the model from ``val_df_full``.

        flagger
            A :class:`TrainingFlag` instance that computes the true good/bad quality flag from ``val_df_full``.
        """
        is_good_true, is_good_pred = self._predict(model, val_df_full, features, flagger)
        return self.metric(is_good_true, is_good_pred)
    
    @staticmethod
    def _predict(model: DecisionTreeClassifier, val_df_full: pd.DataFrame, features: FeatureSet, flagger: TrainingFlag):
        """Apply the model to the given data and predict the quality flags.
        """
        val_df = features.subset_df(val_df_full)
        is_good_true = flagger.make_training_flag(val_df_full).to_numpy()
        is_good_pred = model.predict(val_df.to_numpy())
        return is_good_true, is_good_pred
        
    def metric(self, true_flags: np.ndarray, pred_flags: np.ndarray) -> float:
        """Given the true and predicted flags, return a value representing the quality of the model.
        
        Parameters
        ----------
        true_flags
            Actual quality flags as computed by a :class:`TrainingFlag` instance.

        pred_flags
            Quality flags predicted by the model.

        Returns
        -------
        float
            The quality metric.
        """
        raise NotImplementedError('metric')
        
    @property
    def label(self) -> str:
        """Return a label for this metric, suitable for plots
        """
        raise NotImplementedError('label')
        
    @property
    def extra_metric_names(self) -> Sequence[str]:
        """Returns a list of keys that will be in the dictionary returned by
        ``calc_extra_fields``.
        """
        return tuple()
    
    def calc_extra_metrics(self, model: DecisionTreeClassifier, val_df_full: pd.DataFrame, features: FeatureSet, flagger: TrainingFlag) -> dict:
        """Returns a dictionary with additional metrics for this model that can provide
        more insight into its performance. The keys of the dict must match the list returned
        by ``extra_fields`.

        Parameters
        ----------        
        model
            A trained model.

        val_df_full
            A dataframe containing both the AIRS & CrIS XPAN800 values and the input features
            used by the model.

        flagger
            A :class:`TrainingFlag` instance that computes the true good/bad quality flag from ``val_df_full``.

        Returns
        -------
        dict
            A dictionary with extra metrics (as floats).
        """
        return dict()


class TruthRateMetric(PerformanceMetric):
    """A performance metric that reports the rate of true predictions per total predictions.
    """
    def metric(self, true_flags: np.ndarray, pred_flags: np.ndarray) -> float:
        arrs = self.index_arrays(true_flags, pred_flags)
        n_true = arrs['true_good'].sum() + arrs['true_bad'].sum()
        n_false = arrs['false_good'].sum() + arrs['false_bad'].sum()
        return n_true / (n_true + n_false)
    
    @property
    def label(self):
        return 'Truth rate (0 to 1)'
    
    @staticmethod
    def index_arrays(true_flags: np.ndarray, pred_flags: np.ndarray):
        """Returns a dictionary with logical index arrays specifying which
        flags are correctly and incorrectly good and bad.

        Parameters
        ----------
        true_flags
            Actual quality flags as computed by a :class:`TrainingFlag` instance.

        pred_flags
            Quality flags predicted by the model.
        """
        return dict(
            true_good=true_flags & (true_flags == pred_flags),
            true_bad=~true_flags & (true_flags == pred_flags),
            false_good=pred_flags & (pred_flags != true_flags),
            false_bad=~pred_flags & (pred_flags != true_flags)
        )
    
    @property
    def extra_metric_names(self):
        return ['true_good', 'true_bad', 'false_good', 'false_bad']
    
    def calc_extra_metrics(self, model, val_df_full: pd.DataFrame, features: FeatureSet, flagger: TrainingFlag) -> dict:
        """Returns a dictionary with additional metrics for this model that can provide
        more insight into its performance. For this class, the extra metrics are the fraction
        of soundings that are correctly good ("true_good"), correctly bad ("true_bad"),
        incorrectly good ("false_good"), and incorrectly bad ("false_bad").
        """
        is_good_true, is_good_pred = self._predict(model, val_df_full, features, flagger)
        arrs = self.index_arrays(is_good_true, is_good_pred)
        tot = val_df_full.shape[0]
        return {k: v.sum() / tot for k, v in arrs.items()}
