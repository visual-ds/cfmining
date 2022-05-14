# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
@license: BSD 3-clause.
"""
import numpy as np
import sklearn

def mean_plus_dev_error(y_ref, y_pred, dev=2):
    err = abs(y_ref-y_pred)
    return np.mean(err)+dev*np.std(err)

def mean_error(y_ref, y_pred):
    err = abs(y_ref-y_pred)
    return np.mean(err)

def metric(clf, X, y):
    return mean_error(y, clf.predict_proba(X)[:,1])

def replace(X, X_rep, col):
    newX = X.copy()
    newX[:, col] = X_rep[:, col]
    return newX

def calc_imp(X, y, individual, clf, action_set, repetitions=1):
    X_base = np.concatenate([X.values for i in range(repetitions)], axis=0)
    grid_ = action_set.feasible_grid(individual, return_actions=False, return_percentiles=False, return_immutable=True)
    #X_rep = np.array([[np.random.choice(action._grid) for action in action_set] for i in range(X_base.shape[0])])
    X_rep = np.array([[np.random.choice(grid_[action.name]) for action in action_set] for i in range(X_base.shape[0])])
    importance = np.array([metric(clf,
                                  replace(X_base, X_rep, col),
                                  clf.predict_proba(X_base)[:,1])
                           for col in range(X.shape[1])])
    return importance


class GeneralClassifier():
    """Wrapper to general type of classifer.
    It estimates the importance of the features and assume the classifier as non-monotone.
    
    Parameters
    ----------
    
    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    """
    def __init__(self, classifier, X=None, y=None, metric=None, threshold=0.5):
        self.clf = classifier
        self.threshold = threshold
        from eli5.sklearn import PermutationImportance
        from sklearn.metrics import roc_auc_score, mean_absolute_error
        def deviat(clf, X, y):
            if metric is None:
                #return mean_absolute_error(y, clf.predict_proba(X)[:,1])
                return roc_auc_score(y, clf.predict_proba(X)[:,1])
            else:
                return metric(y, clf.predict_proba(X)[:,1])
        perm = PermutationImportance(self.clf, scoring=deviat, n_iter=10).fit(X, y)
        #perm = PermutationImportance(self.clf, scoring=deviat).fit(X, classifier.predict_proba(X)[:,1])
        self.importances = abs(perm.feature_importances_)

    @property
    def feat_importance(self):
        return self.importances

    @property
    def monotone(self):
        return False
    
    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.predict_proba(value)>=self.threshold:
            return True
        else:
            return False

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
        return self.clf.predict_proba([value])[0, 1]


class MonotoneClassifier(GeneralClassifier):
    """Wrapper to general type of classifer.
    It estimates the importance of the features and assume the classifier as monotone.
    
    Parameters
    ----------
    
    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    """
    @property
    def monotone(self):
        return True

class LinearClassifier():
    """Wrapper to linear classifers.
    Calculates the importance using the coeficients from linear classification and assume the classifier as monotone.
    
    Parameters
    ----------
    
    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    """
    def __init__(self, classifier, X=None, y=None, metric=None, threshold=0.5):
        self.clf = classifier
        self.threshold = threshold
        if type(self.clf) is sklearn.pipeline.Pipeline:
            try:
                coeficients = self.clf['clf'].coef_[0]/self.clf['std'].scale_
            except KeyError:
                print('sdadsa')
                coeficients = self.clf['clf'].coef_[0]
        else:
            coeficients = self.clf.coef_[0]
        
        self.importances = abs(coeficients*X.std(axis=0))

    @property
    def feat_importance(self):
        return self.importances

    @property
    def monotone(self):
        return True
    
    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.predict_proba(value)>=self.threshold:
            return True
        else:
            return False

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
        return self.clf.predict_proba([value])[0, 1]


class LinearRule():
    def __init__(self, coef, Xtrain, threshold=0):
        self.coef = coef
        self.threshold = threshold
        self.Xtrain = Xtrain

    @property
    def feat_importance(self):
        return abs(self.coef*self.Xtrain.mean(axis=0))

    @property
    def monotone(self):
        return True
    
    def predict(self, value):
        """Predicts if the probability is higher than threshold"""
        if self.coef@value>=self.threshold:
            return True
        else:
            return False
            
    def predict_proba(self, value):
        """Predicts if the probability is higher than threshold"""
        return self.coef@value

from cfmining.predictors_utils import TreeExtractor

class TreeClassifier(GeneralClassifier, TreeExtractor):
    """Wrapper tree based classifiers.
    It extracts the information from the branching to speed-up the prediction based on the reference sample.
    It estimates the importance of the features and assume the classifier as non-monotone 
    but explores the structure of the tree to calculate the maximal probability of a branch in the optimization procedure.
    
    Parameters
    ----------
    
    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    use_predict_max : bool,
        Set if the wrapper is allowed to calculate the maximal probability of a optimization branch.
    clf_type : float (sklearn or lightgbm),
        Family of tree classifier.
    """
    def __init__(self, classifier,
                 X=None, y=None, metric=None, threshold=0.5,
                 use_predict_max=False, clf_type='sklearn'):
 
        super().__init__(classifier, X, y, metric, threshold)
        self.clf = classifier
        self.clf_type = clf_type
        self.threshold = threshold
        self.use_predict_max = use_predict_max

    def fit(self, individual, action_set):
        self.names = list(action_set.df['name'])
        grid_ = action_set.feasible_grid(individual, return_actions=False,
                                         return_percentiles=False, return_immutable=True)
        grid_ = {idx:grid_[name] for idx, name in enumerate(grid_)}
        self.extract_tree(grid_)

    def predict_proba(self, value):
        """Calculates probability of achieving desired classification."""
        n_estimators = len(self.forest)
        prediction = 0
        for leaves_tree in self.forest:
            for leaf in leaves_tree:
                for v, name in zip(value[leaf['used_features']], leaf['used_features']):
                    if v not in leaf['variables'][name]:
                        break
                else:
                    prediction+=leaf['prediction']
                    break
        if self.clf_type=='sklearn':
            return prediction/self.n_estimators
        elif self.clf_type=='lightgbm':
            return 1/(1+np.exp(-prediction))

    def predict_max_(self, value, fixed_vars):
        """Calculates the maximal probability of a optimization branch."""
        n_estimators = len(self.forest)
        prediction = 0
        for leaves_tree in self.forest:
            prob = self.lowest_value
            for leaf in leaves_tree:
                for v, name in zip(value[fixed_vars], fixed_vars):
                    if v not in leaf['variables'][name]:
                        break
                else:
                    prob = max(prob, leaf['prediction'])
            prediction+=prob
        if self.clf_type=='sklearn':
            return prediction/self.n_estimators
        elif self.clf_type=='lightgbm':
            return 1/(1+np.exp(-prediction))

    def predict_max(self, value, fixed_vars):
        """Calculates the maximal probability of a optimization branch."""
        fixed = set(fixed_vars)
        n_estimators = len(self.forest)
        prediction = 0
        for leaves_tree in self.forest:
            prob = -np.inf
            for leaf in leaves_tree:
                feat_u = list(fixed.intersection(leaf['used_features']))
                for v, name in zip(value[feat_u], feat_u):
                    if v not in leaf['variables'][name]:# and name in fixed_vars:
                        break
                else:
                    prob = max(prob, leaf['prediction'])
            prediction+=prob
        if self.clf_type=='sklearn':
            return prediction/self.n_estimators
        elif self.clf_type=='lightgbm':
            return 1/(1+np.exp(-prediction))

class MonotoneTree(TreeClassifier):
    """Wrapper tree based classifiers.
    It extracts the information from the branching to speed-up the prediction based on the reference sample.
    It estimates the importance of the features and assume the classifier as monotone it should only be used with lightgbm with monotonicity constraint.
    but explores the structure of the tree to calculate the maximal probability of a branch in the optimization procedure.
    
    Parameters
    ----------
    
    classifier : sklearn type classifer,
        General classifier with predict_proba method.
    X : numpy array,
        Input Samples
    y : numpy array,
        Output Samples.
    metric : function,
        User specific metric function to estimate importance.
    threshold : float,
        User defined threshold for the classification.
    use_predict_max : bool,
        Set if the wrapper is allowed to calculate the maximal probability of a optimization branch.
    clf_type : float (sklearn or lightgbm),
        Family of tree classifier.
    """
    @property
    def monotone(self):
        return True

'''
class TreeClassifier(GeneralClassifier):
    def __init__(self, classifier, individual, action_set, 
                 X=None, y=None, metric=None, threshold=0.5,
                 use_predict_max=False, general=None):
        from actionsenum.mip_algorithms import recursive_tree_
 
        self.clf = classifier
        self.threshold = threshold

        self.names = list(action_set.df['name'])
        grid_ = action_set.feasible_grid(individual, return_actions=False, return_percentiles=False, return_immutable=True)
        leaves = [recursive_tree_(tree_, self.names, grid_, 0, tree_name=i)
                  for i, tree_ in enumerate(classifier.estimators_)]

        for leaves_tree in leaves:
            for leaf in leaves_tree:
                leaf['variables'] = {var:set(leaf['variables'][var]) for var in leaf['variables']}
                leaf['used_feat'] = list(leaf['used_feat'])[::-1]
                leaf['used_idx'] = [self.names.index(name) for name in leaf['used_feat']]

        self.leaves = leaves

        if general is not None:
            self.importances = general.importances
        else:
            self.importances = classifier.feature_importances_
        self.use_predict_max = use_predict_max

    def predict_proba(self, value):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            for leaf in leaves_tree:
                active_leaf = True
                for v, name in zip(value[leaf['used_idx']], leaf['used_feat']):
                    if v not in leaf['variables'][name]:
                        active_leaf=False
                        break
                if active_leaf:
                    prediction+=leaf['prediction']
                    break
        return prediction/n_estimators

    def predict_max(self, value, fixed_vars):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            prob = 0
            for leaf in leaves_tree:
                active_leaf = True
                for v, idx, name in zip(value[leaf['used_idx']], leaf['used_idx'], leaf['used_feat']):
                    if v not in leaf['variables'][name] and idx in fixed_vars:
                        active_leaf=False
                        break
                if active_leaf:
                    prob = max(prob, leaf['prediction'])
            prediction+=prob
        #print(fixed_vars, prediction/n_estimators)
        return prediction/n_estimators


class TreeClassifier2(GeneralClassifier):
    def __init__(self, classifier,
                 X=None, y=None, metric=None, threshold=0.5,
                 use_predict_max=False):
 
        super().__init__(classifier, X, y, metric, threshold)
        self.clf = classifier
        self.threshold = threshold
        self.use_predict_max = use_predict_max

    def fit(self, individual, action_set):
        from actionsenum.mip_algorithms import recursive_tree_
        self.names = list(action_set.df['name'])
        grid_ = action_set.feasible_grid(individual, return_actions=False, return_percentiles=False, return_immutable=True)
        leaves = [recursive_tree_(tree_, self.names, grid_, 0, tree_name=i)
                  for i, tree_ in enumerate(self.clf.estimators_)]

        for leaves_tree in leaves:
            for leaf in leaves_tree:
                leaf['variables'] = {var:set(leaf['variables'][var]) for var in leaf['variables']}
                leaf['used_feat'] = set(leaf['used_feat'])
                leaf['used_idx'] = [self.names.index(name) for name in leaf['used_feat']]

        self.leaves = leaves

    def predict_proba(self, value):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            for leaf in leaves_tree:
                active_leaf = True
                for v, name in zip(value[leaf['used_idx']], leaf['used_feat']):
                    if v not in leaf['variables'][name]:
                        active_leaf=False
                        break
                if active_leaf:
                    prediction+=leaf['prediction']
                    break
        return prediction/n_estimators

    def predict_max(self, value, fixed_vars):
        n_estimators = len(self.leaves)
        prediction = 0
        for leaves_tree in self.leaves:
            prob = 0
            for leaf in leaves_tree:
                active_leaf = True
                for v, idx, name in zip(value[leaf['used_idx']], leaf['used_idx'], leaf['used_feat']):
                    if v not in leaf['variables'][name] and idx in fixed_vars:
                        active_leaf=False
                        break
                if active_leaf:
                    bla = leaf['tree'], leaf['name']
                    prob = max(prob, leaf['prediction'])
            prediction+=prob
        return prediction/n_estimators
'''
