# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
"""
import numpy as np
import pandas as pd
import mip
import copy

from cfmining.criteria import NonDomCriterion
from cfmining.mip_builder import _RecourseBuilderCPX, RecourseBuilder

class LinearRecourseActions():
    def __init__(self, action_set, pivot, 
                 coefficients, intercept, threshold, max_changes=None):
        
        self.coefficients = coefficients
        self.intercept = intercept
        self.threshold = threshold
        self.max_changes = max_changes
        
        self.action_set = action_set
        self.pivot = pivot
    
    def fit(self):
        p = self.threshold
        x = self.pivot
        action_set = self.action_set
        
        
        rb = RecourseBuilder(
              optimizer="cplex",
              coefficients=self.coefficients,
              intercept=self.intercept-(np.log(p / (1. - p))),
              action_set=action_set,
              x=x
        )

        if self.max_changes is not None:
            rb.max_items = self.max_changes
        rb.x = x
        build_info, indices = rb._get_mip_build_info()
        output_1 = rb.fit()
        self.stats = output_1
        self.recourse = rb
        
        solution = output_1['actions']+self.pivot
        
        self.solution = solution

class LinearRecourseActionsMulti():
    def __init__(self, action_set, pivot, 
                 coefficients, intercept, threshold, max_changes=None, compare=None, enumeration_type='remove_dominated'):
        #remove_dominated, remove_number_actions
        
        self.coefficients = coefficients
        self.intercept = intercept
        self.threshold = threshold
        self.max_changes = max_changes
        
        self.action_set = action_set
        self.pivot = pivot
        
        if compare is None:
            self.names = list(action_set.df['name'])
            self.feat_direction = np.array([action_set[feat_name].flip_direction
                                            for feat_name in self.names])
            self.compare = NonDomCriterion(self.feat_direction)
        else:
            self.compare = compare

        self.enumeration_type = enumeration_type

    def nearest_action(self, solution, grid_):
        return np.array([grid_[name][np.abs(solution[idx] - grid_[name]).argmin()] for idx, name in enumerate(grid_)])

    def fit(self):
        p = self.threshold
        x = self.pivot
        action_set = self.action_set
        grid_ = self.action_set.feasible_grid(x, return_actions=False, return_percentiles=False, return_immutable=True)
        
        
        rb = RecourseBuilder(
              optimizer="cplex",
              coefficients=self.coefficients,
              intercept=self.intercept-(np.log(p / (1. - p))),
              action_set=action_set,
              mip_cost_type='total' if self.enumeration_type=='remove_dominated' else 'max',
              #x=x,
        )
        #print(rb.super()._actionable_indices)
        if self.max_changes is not None:
            rb.max_items = self.max_changes
        rb.x = x
        rb.print_flag = False
        #build_info, indices = rb._get_mip_build_info()
        output = rb.populate(float('inf'), enumeration_type=self.enumeration_type)
        self.recourse = rb
        
        self.solutions = [self.nearest_action(out['actions']+self.pivot, grid_)
                          for out in output]
        
        solutions = []
        '''
        for i,solution in enumerate(self.solutions):
            optimal = True
            for j,comp_sol in enumerate(self.solutions):
                if j>i and self.compare.greater_than(solution, comp_sol):
                    if not self.compare.greater_than(comp_sol, solution) or j>i:
                        optimal = False
                        break
            if optimal:
                solutions += [solution]
        '''
        for i,solution in enumerate(self.solutions):
            #print(self.compare.f(solution))
            optimal = True
            for j,comp_sol in enumerate(self.solutions):
                if self.compare.greater_than(solution, comp_sol):
                    if i<j or not self.compare.greater_than(comp_sol, solution):
                        optimal = False
                        break
            if optimal:
                solutions += [solution]
        self.solutions = solutions
'''
def recursive_tree_(model, feat_names, grid, node, tree_name='', used_feat=set()):
    from sklearn.tree import _tree
    tree_name = tree_name
    #print('node', tree_name, node)
    
    feature = model.tree_.feature[node]
    name = feat_names[feature] if feature!= _tree.TREE_UNDEFINED else None
    
    if name is None:
        leaf_info = {}
        leaf_info['tree'] = tree_name
        leaf_info['name'] = node
        leaf_info['weight'] = 1
        leaf_info['variables'] = grid
        leaf_info['prediction'] = model.tree_.value[node][0][1]/float(sum(model.tree_.value[node][0]))
        leaf_info['used_feat'] = used_feat
        #print(model.tree_.value[node], leaf_info['prediction'])
        return [leaf_info]
    
    threshold = model.tree_.threshold[node]
    left = grid[name]<=threshold
    
    returns = []
    if sum(left)!=0:
        grid_left = copy.copy(grid)
        grid_left[name] = grid_left[name][left]
        feat_left = copy.copy(used_feat)
        feat_left.add(name)
        returns += recursive_tree_(model, feat_names, grid_left, model.tree_.children_left[node], tree_name, feat_left)
    
    if sum(~left)!=0:
        grid_right = copy.copy(grid)
        grid_right[name] = grid_right[name][~left]
        feat_right = copy.copy(used_feat)
        feat_right.add(name)
        returns += recursive_tree_(model, feat_names, grid_right, model.tree_.children_right[node], tree_name, feat_right)

    return returns
'''

from cfmining.predictors_utils import TreeExtractor

class ForestRecourseActions():
    def __init__(self, action_set, pivot, 
                 classifier, clf_type='sklearn', threshold=0.5,
                 max_changes = None, compare=None, clean_suboptimal=False, warm_solutions=None,
                 full_grid=False, max_seconds=float('inf'), max_gap=0.0, cost_type='max', multi_solution=False):
        
        self.names = list(action_set.df['name'])
        self.idxs = np.arange(len(self.names))
        self.pivot = pivot
        #self.name2ind = {feat_name:i for i,feat_name in enumerate(self.names)}
        self.max_changes = max_changes
        self.clf = classifier
        self.action_set = action_set
        self.threshold = threshold
        self.clf = classifier
        self.clf_type = clf_type
        #self.estimators = self.clf.estimators_ if hasattr(self.clf, 'estimators_') else [self.clf]
        #self.n_estimators = len(self.estimators)
        
        self.full_grid = full_grid
        self.pivot_percentiles = {feat_name: self.action_set[feat_name].percentile(pivot[i])
                                  for i, feat_name in enumerate(self.names)}
        
        self.max_seconds = max_seconds
        self.max_gap = max_gap
        self.cost_type = cost_type
        self.multi_solution = multi_solution

        if compare is None:
            self.names = list(action_set.df['name'])
            self.feat_direction = np.array([action_set[feat_name].flip_direction
                                            for feat_name in self.names])
            self.compare = NonDomCriterion(self.feat_direction)
        else:
            self.compare = compare
        
        
    def cost(self, feat_name, value):
        return abs(self.action_set[feat_name].percentile(value)-self.pivot_percentiles[feat_name])        
        
    def build_model(self):
        d = len(self.action_set)
        #treeActions.feasible_grid(self.pivot, return_actions=False, return_percentiles=False)
        if self.full_grid:
            grid_ = self.action_set.full_grid()
        else:
            grid_ = self.action_set.feasible_grid(self.pivot, return_actions=False, return_percentiles=False, return_immutable=True)
        grid_ = {idx:grid_[name] for idx, name in enumerate(grid_)}

        self.forest = TreeExtractor(self.clf, clf_type=self.clf_type).extract_tree(grid_, return_forest=True)
        self.n_estimators = len(self.forest)
        
        self.costs = {feature:{value:self.cost(self.names[feature], value)
                                 for value in grid_[feature]}
                      for feature in grid_}
        
        #self.leaves = [leaf for i, tree_ in enumerate(self.estimators)
        #               for leaf in recursive_tree_(tree_, self.names, grid_, 0, tree_name=i)]
        
        self.mip_model = mip.Model(solver_name='gurubi')
        
        self.mip_var_bins = {feature:{value:
                                        self.mip_model.add_var(var_type=mip.BINARY, name='%s=%f'%(feature, value))
                                        for value in grid_[feature]}
                             for feature in grid_}
        self.mip_leaves = {tree_name:{leaf['name']:self.mip_model.add_var(var_type=mip.BINARY, name='%s=%f'%(leaf['tree'], leaf['name']))
                                      for leaf in tree}
                           for tree_name, tree in enumerate(self.forest)}
        
        for tree in self.forest:
            for leaf in tree:
                expr = mip.xsum(1/len(leaf['used_features']) * self.mip_var_bins[feature][value]
                                for feature in leaf['used_features']
                                for value in leaf['variables'][feature])
                self.mip_model += self.mip_leaves[leaf['tree']][leaf['name']] <= expr
            
        ## all feature has at least one variable activated
        for feature in self.idxs:
            expr = mip.xsum(self.mip_var_bins[feature][value]
                            for value in self.mip_var_bins[feature])
            self.mip_model += expr==1

        if self.max_changes is not None:
            expr = mip.xsum(self.mip_var_bins[feature][value]
                            for feature in self.idxs
                            for value in self.mip_var_bins[feature]
                            if self.pivot[feature]!=value)
            self.mip_model += expr<=self.max_changes
            
        
        for tree_name in range(self.n_estimators):
            expr = mip.xsum(self.mip_leaves[tree_name][leaf_name] for leaf_name in self.mip_leaves[tree_name])
            self.mip_model += expr==1
        
        if self.clf_type=='sklearn':
            expr = mip.xsum(leaf['weight']*leaf['prediction']/self.n_estimators*self.mip_leaves[leaf['tree']][leaf['name']] for tree in self.forest for leaf in tree)
            self.mip_model += expr>=self.threshold
        elif self.clf_type=='lightgbm':
            expr = mip.xsum(leaf['weight']*leaf['prediction']*self.mip_leaves[leaf['tree']][leaf['name']] for tree in self.forest for leaf in tree)
            self.mip_model += expr>=-np.log((1. - self.threshold)/self.threshold)
        
        if self.cost_type =='linear':
            obj = mip.xsum(self.costs[feature][value]*self.mip_var_bins[feature][value]
                           for feature in self.idxs
                           for value in grid_[feature])
        elif self.cost_type =='max':
            obj = self.mip_model.add_var(var_type=mip.CONTINUOUS, name='obj')
            for feature in self.idxs:
                self.mip_model += obj>= mip.xsum(self.costs[feature][value]*self.mip_var_bins[feature][value]
                                                 for value in grid_[feature])

        self.mip_model.objective = mip.minimize(obj)
        
    def fit(self, max_seconds=None, max_gap=None, threads=-1):
        self.mip_model.max_gap = max_gap if max_gap is not None else self.max_gap
        self.mip_model.threads = threads
        self.solutions = []
        direc = [action.step_direction for action in self.action_set]
        self.solution = np.full(len(self.action_set), np.nan)
        self.solutions = []
        while True:
            self.mip_model.optimize(max_seconds=max_seconds if max_seconds is not None else self.max_seconds)
            if not self.mip_model.status==mip.OptimizationStatus.OPTIMAL:
                break
            
            solution = np.array([sum([value 
                                      for value, var in zip(self.mip_var_bins[feature], self.mip_var_bins[feature].values())
                                      if var.x >= 0.99
                                      ])
                                 for feature in self.idxs])
            if self.multi_solution:
                self.solutions.append(solution)
                expr = mip.xsum(var
                                 for feature in self.idxs
                                 for value, var in zip(self.mip_var_bins[feature],
                                                       self.mip_var_bins[feature].values())
                                 if direc[feature]*value>=direc[feature]*solution[feature])
                self.mip_model += expr <= len(solution)-1
            else:
                self.solution = solution
                break

        if self.multi_solution:
            solutions = []
            for i,solution in enumerate(self.solutions):
                #print(self.compare.f(solution))
                optimal = True
                for j,comp_sol in enumerate(self.solutions):
                    if self.compare.greater_than(solution, comp_sol):
                        if i<j or not self.compare.greater_than(comp_sol, solution):
                            optimal = False
                            break
                if optimal:
                    solutions += [solution]
            self.solutions = solutions
