import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .binary_flag import metrics
from . import features

class RfeQuality:
    def __init__(self, metric: metrics.PerformanceMetric, val_df: pd.DataFrame, flagger: metrics.TrainingFlag):
        self.metric = metric
        self.val_df = val_df
        self.flagger = flagger
        
    def __call__(self, model):
        return self.report_performance(model)['performance']
        
    def report_performance(self, model: DecisionTreeClassifier, feature_names=None):
        if feature_names is None:
            feature_names = model.feature_names_in_
        selector = features.ColnameFeatureSet(feature_names)
        performance = self.metric.validate(model=model, val_df_full=self.val_df, features=selector, flagger=self.flagger)
        extra_data = self.metric.calc_extra_metrics(model=model, val_df_full=self.val_df, features=selector, flagger=self.flagger)
        extra_data['performance'] = performance
        return extra_data


def recursive_feature_elimination(
    training_df_full: pd.DataFrame,
    val_df_full: pd.DataFrame,
    input_features: features.FeatureSet,
    flagger: metrics.TrainingFlag,
    quality_metric: metrics.PerformanceMetric = metrics.TruthRateMetric()
):
    current_features = list(input_features.subset_df(training_df_full).columns)
    training_flag = flagger.make_training_flag(training_df_full).to_numpy()
    importance = RfeQuality(metric=quality_metric, val_df=val_df_full, flagger=flagger)
    
    
    feature_ranks = []
    perf_dicts = []
    while len(current_features) > 1:
        current_perf = _get_model_performance(
            current_features,
            training_df_full=training_df_full,
            training_flags=training_flag,
            importance=importance,
            model_factory=DecisionTreeClassifier
        )
        
        feature_to_remove = _eliminate_feature(
            current_features=current_features,
            training_df_full=training_df_full,
            training_flags=training_flag,
            importance=importance,
            model_factory=DecisionTreeClassifier
        )
        
        feature_ranks.append(feature_to_remove)
        perf_dicts.append(current_perf)
        current_features.remove(feature_to_remove)
        
    return pd.DataFrame(perf_dicts, index=feature_ranks)
    
    
def _eliminate_feature(current_features, training_df_full, training_flags, importance, model_factory):
    
    least_important_feature = None
    best_perf_dict = None
    
    n = len(current_features)
    for i, feat in enumerate(current_features):
        print(f'\r[{i+1}/{n}] Testing without {feat:32}', end='')
        test_features = [f for f in current_features if f != feat]
        this_perf_dict = _get_model_performance(test_features, training_df_full, training_flags, importance, model_factory)
        
        if least_important_feature is None:
            least_important_feature = feat
            best_perf_dict = this_perf_dict
        elif this_perf_dict['performance'] > best_perf_dict['performance']:
            least_important_feature = feat
            best_perf_dict = this_perf_dict
            
    print(f'\n  -> Removing {least_important_feature}')
    return least_important_feature

def _get_model_performance(feat_list, training_df_full, training_flags, importance, model_factory):
    feat_getter = features.ColnameFeatureSet(feat_list)
    train_X = feat_getter.subset_df(training_df_full).to_numpy()
    model = model_factory()
    model.fit(train_X, training_flags)
    return importance.report_performance(model, feature_names=feat_list)