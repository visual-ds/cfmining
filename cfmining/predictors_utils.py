import copy
from sklearn.tree import _tree
import numpy as np

class Setr():
    def __init__(self, lb=-float('inf'), ub=float('inf'), openLb=False, openUb=False):
        if lb>ub:
            lb=ub
            openLb=True
        self.lb = lb
        self.ub = ub
        self.openLb = openLb if lb!=-float('inf') else True
        self.openUb = openUb if ub!=float('inf') else True
        self.greater = np.greater if openLb else np.greater_equal
        self.less = np.less if openUb else np.less_equal
        if lb==-float('inf') and ub==float('inf'):
            self.__acontains__ = self.i
        elif lb==-float('inf'):
            self.__acontains__ = self.g
        elif ub==float('inf'):
            self.__acontains__ = self.l
        else:
            self.__acontains__ = self.b
        
    def i(self, value):
        return True

    def g(self, value):
        return self.greater(value, self.lb) 

    def l(self, value):
        return self.less(value, self.ub)

    def b(self, value):
        return self.greater(value,self.lb) and self.less(value, self.ub)

    def __contains__(self, value):
        return self.__acontains__(value)
    
    #def __contains__(self, value):
    #    return self.greater(value,self.lb) and self.less(value, self.ub)

    # intersection
    def intersection(self, item):
        if self.lb==-float('inf') and self.ub==float('inf'):
            return copy.copy(item)
        if type(item)==Setr:
            lb = max(self.lb, item.lb)
            openLb = self.openLb if lb==self.lb else item.openLb
            openLb = self.openLb and item.openLb if self.lb==item.openLb else openLb

            ub = min(self.ub, item.ub)
            openUb = self.openUb if ub==self.ub else item.openUb
            openUb = self.openUb and item.openUb if self.ub==item.openUb else openUb

            ans = Setr(lb, ub, openLb, openUb)
        elif type(item)==set:
            ans = set([i for i in item if i in self])
        return ans
 
    def __repr__(self):
        return (("(" if self.openLb else "[")+
                str(self.lb)+
                ","+
                str(self.ub)+
                (")" if self.openUb else "]"))
    @property
    def empty(self):
        if self.lb>self.ub:
            return True
        elif self.lb==self.ub and (self.openLb or self.openUb):
            return True
        else:
            return False

def extract_sklearn(model, grid, node=0, tree_name='', used_features=set()):
    tree_name = tree_name
    
    feature = model.tree_.feature[node] if model.tree_.feature[node]!= _tree.TREE_UNDEFINED else None
    
    if feature is None:
        leaf_info = {}
        leaf_info['tree'] = tree_name
        leaf_info['name'] = node
        leaf_info['weight'] = 1
        leaf_info['variables'] = grid
        leaf_info['prediction'] = model.tree_.value[node][0][1]/float(sum(model.tree_.value[node][0]))
        leaf_info['used_features'] = used_features
        return [leaf_info]
   

    threshold = model.tree_.threshold[node]
    is_left = grid[feature]<=threshold
    left = grid[feature][is_left]
    right = grid[feature][~is_left]
    
    returns = []
    if left.size!=0:
        grid_left = copy.copy(grid)
        grid_left[feature] = left

        features_left = copy.copy(used_features)
        features_left.add(feature)

        left_node = model.tree_.children_left[node]

        returns += extract_sklearn(model, grid_left, left_node, tree_name, features_left)
    
    if right.size!=0:
        grid_right = copy.copy(grid)
        grid_right[feature] = right

        features_right = copy.copy(used_features)
        features_right.add(feature)

        right_node = model.tree_.children_right[node]

        returns += extract_sklearn(model, grid_right, right_node, tree_name, features_right)

    return returns

def extract_lightgbm(model, tree_, grid, tree_name='', used_features=set()):
    tree_name = tree_name
    #print(tree_.keys())
    feature = tree_['split_feature'] if 'split_feature' in tree_ else None
    
    if feature is None:
        leaf_info = {}
        leaf_info['tree'] = tree_name
        leaf_info['name'] = tree_['leaf_index']
        leaf_info['weight'] = 1
        leaf_info['variables'] = grid
        leaf_info['prediction'] = tree_['leaf_value']
        leaf_info['used_features'] = used_features
        return [leaf_info]
    
    threshold = tree_['threshold']
    is_left = grid[feature]<=threshold
    left = grid[feature][is_left]
    right = grid[feature][~is_left]
    
    returns = []
    if left.size!=0:
        grid_left = copy.copy(grid)
        grid_left[feature] = left

        features_left = copy.copy(used_features)
        features_left.add(feature)

        left_node = tree_['left_child']

        returns += extract_lightgbm(model, left_node, grid_left, tree_name, features_left)
    
    if right.size!=0:
        grid_right = copy.copy(grid)
        grid_right[feature] = right

        features_right = copy.copy(used_features)
        features_right.add(feature)

        right_node = tree_['right_child']

        returns += extract_lightgbm(model, right_node, grid_right, tree_name, features_right)

    return returns

class TreeExtractor():
    def __init__(self, clf, clf_type='sklearn'):
        if clf_type=='sklearn':
            from sklearn import ensemble
            from sklearn import tree
            assert type(clf) in [ensemble.ExtraTreesClassifier,
                                 ensemble.RandomForestClassifier,
                                 tree.DecisionTreeClassifier,
                                 tree.ExtraTreeClassifier], 'Not a sklearn classifier'
        elif clf_type=='lightgbm':
            import lightgbm
            assert type(clf)==lightgbm.LGBMClassifier, 'Not a lightgbm classifier'
        self.clf_type = clf_type
        self.clf = clf

    def extract_tree(self, grid, return_forest=False):
        if self.clf_type=='sklearn':
            self.estimators = self.clf.estimators_ if hasattr(self.clf, 'estimators_') else [self.clf]
            self.n_estimators = len(self.estimators)
            self.forest = [extract_sklearn(tree_, grid, tree_name=i)
                                for i, tree_ in enumerate(self.estimators)]
        elif self.clf_type=='lightgbm':
            self.estimators = self.clf.booster_.dump_model()['tree_info']
            self.n_estimators = len(self.estimators)
            self.forest = [extract_lightgbm(self.clf, tree_['tree_structure'], grid, tree_name=i)
                                for i, tree_ in enumerate(self.estimators)]
        self.lowest_value = np.inf
        for tree in self.forest:
            for leaf in tree:
                vars_ = sorted(leaf['variables'], key=lambda var: len(leaf['variables'][var])/len(grid[var]))
                leaf['variables'] = {var:set(leaf['variables'][var]) for var in vars_}
                leaf['used_features'] = list(leaf['used_features'])
                self.lowest_value = min(self.lowest_value, leaf['prediction'])
        if return_forest:
            return self.forest
'''
    def predict_proba(self, value):
        prediction = 0
        for tree in self.forest:
            for leaf in tree:
                active_leaf = True
                for feature, v  in zip(leaf['used_features'], value[leaf['used_features']]):
                    if v not in leaf['variables'][feature]:
                        active_leaf=False
                        break
                if active_leaf:
                    prediction+=leaf['prediction']
                    break
        if self.clf_type=='sklearn':
            return prediction/self.n_estimators
        elif self.clf_type=='lightgbm':
            return 1/(1+np.exp(-prediction))

    def predict_max(self, value, fixed_vars):
        prediction = 0
        for tree in self.forest:
            prob = 0
            for leaf in tree:
                active_leaf = True
                for feature, v  in zip(leaf['used_features'], value[leaf['used_features']]):
                    if v not in leaf['variables'][feature] and feature in fixed_vars:
                        active_leaf=False
                        break
                if active_leaf:
                    prob = max(prob, leaf['prediction'])
            prediction+=prob
        #if self.clf_type=='sklearn':
        return prediction/self.n_estimators
        #elif self.clf_type=='lightgbm':
        #    return 1/(1+np.exp(-prediction)) 
'''
