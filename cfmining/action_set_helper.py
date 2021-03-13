import warnings
import numpy as np

def parse_sklearn_forest(model, feat_names):
    thresholds = None
    for tree in model.estimators_:
        thresholds = parse_sklearn_tree_(tree, feat_names, thresholds)
    
    for feat_name in feat_names:
        thresholds[feat_name] = np.array(sorted(set(thresholds[feat_name])))
        
    return thresholds

def parse_sklearn_tree_(model, feat_names, thresholds = None):
    from sklearn.tree import _tree
    if thresholds is None:
        thresholds = {}
        for feat_name in feat_names:
            thresholds[feat_name] = []

    split_feature = [
        feat_names[i] if i != _tree.TREE_UNDEFINED else None
        for i in model.tree_.feature
    ]

    for node in range(model.tree_.node_count):
        name = split_feature[node]
        threshold = model.tree_.threshold[node]
        if name is not None:
            thresholds[name].append(threshold)
    
    return thresholds
        
def parse_sklearn_tree(model, feat_names):
    thresholds = parse_sklearn_tree_(model, feat_names)            
    for feat_name in feat_names:
        thresholds[feat_name] = np.array(sorted(set(thresholds[feat_name])))
        
    return thresholds

def parse_lgb_tree(tree, feat_names, thresholds = None):
    if thresholds is None:
        thresholds = {}
        for feat_name in feat_names:
            thresholds[feat_name] = []
    if 'split_feature' not in tree:
        return
    feat_name = feat_names[tree['split_feature']]
    thresholds[feat_name].append(tree['threshold'])
    parse_lgb_tree(tree['left_child'], feat_names, thresholds=thresholds)
    parse_lgb_tree(tree['right_child'], feat_names, thresholds=thresholds)
    return thresholds

def parse_lgb_forest(model, feat_names):
    tree_infos = model.booster_.dump_model()['tree_info']
    thresholds = None
    for tree_info in tree_infos:
        thresholds = parse_lgb_tree(tree_info['tree_structure'], feat_names, thresholds)
    
    for feat_name in feat_names:
        thresholds[feat_name] = np.array(sorted(set(thresholds[feat_name])))
    return thresholds

def parse_forest(forest, feat_names):
    """
    helper function to parse coefficients and intercept from linear classifier arguments

    *args and **kwargs can contain either:
        - sklearn classifiers with 'coef_' and 'intercept_' fields (keyword: 'clf', 'classifier')
        - vector of coefficients (keyword: 'coefficients')
        - intercept: set to 0 by default (keyword: 'intercept')

    returns:
        w - np.array containing coefficients of linear classifier (finite, flattened)
        t - float containing intercept of linear classifier (finite, float)

    raises:
        ValueError if fails to parse classifier arguments

    :return:
    """
    try:
        from sklearn import tree
        from sklearn import ensemble
    except ImportError as e:
        warnings.warn('Package sklearn is not installed. If you are using such classifier, please install it.')

    if 'tree' in vars() and isinstance(forest, tree.DecisionTreeClassifier):
        thresholds = parse_sklearn_tree(forest, feat_names)
        return thresholds
    
    if 'tree' in vars() and (isinstance(forest, ensemble.RandomForestClassifier) or
                             isinstance(forest, ensemble.ExtraTreesClassifier)):
        thresholds = parse_sklearn_forest(forest, feat_names)
        return thresholds

    try:
        import lightgbm as lgb
    except ImportError as e:
        warnings.warn('Package lightgbm is not installed. If you are using such classifier, please install it.')

    if 'lgb' in vars() and isinstance(forest, lgb.LGBMClassifier):
        thresholds = parse_lgb_forest(forest, feat_names)
        return thresholds
