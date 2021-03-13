# -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
@license: BSD 3-clause.
"""

import numpy as np
import functools

class PercentileCalculator():
    """Class that is capable of calculating the percentile cost
    (change of the feature in percentiles).
    
    Parameters
    ----------
    
    X : numpy array,
        Contains the samples.
    action_set : ActionSet type class,
        Contains the discretization of the features to find the recourses.
     
    """
    def __init__(self, X=None, action_set=None, method='ustun'):
        self.method = method
        if self.method=='ustun':
            assert X is not None or action_set is not None, 'X or action_set should not be None.'
            if action_set is not None:
                self.action_set = action_set
            elif X is None:
                from recourse.builder import ActionSet
                self.action_set = ActionSet(X = X)
                
            self.names = self.action_set.df['name'].values
        self.percentile_vec = np.vectorize(self.percentile)
    
    @functools.lru_cache(maxsize=1024)
    def percentile(self, value, feature):
        """Calculates the percentile."""
        if self.method=='ustun':
            return self.action_set[feature].percentile(value)

    def percVec(self, vec):
        #return self.percentile_vec(vec, self.names)
        return np.array([self.percentile(value, feature) for feature, value in zip(self.names, vec)])

class PercentileCriterion():
    """Class that using the percentile calculator is capable of defining
    which solution have a higher percentile cost.
    
    Parameters
    ----------
    
    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    perc_calc : PercentileCalculator class,
        Percentile calculator for that set of samples.     
    """
    def __init__(self, pivot, perc_calc):
        self.pivot = pivot
        self.perc_calc = perc_calc     
        self.pivotP = self.perc_calc.percVec(pivot)
    
    def f(self, solution):
        solutionP = self.perc_calc.percVec(solution)
        return abs(solutionP-self.pivotP).max()
    
    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        new_obj = self.f(new_sol)
        old_obj = self.f(old_sol)
                    
        return new_obj>=old_obj

class PercentileChangesCriterion(PercentileCriterion):
    """Class that using the percentile calculator and number of changes
    is capable of defining if a solution is worse in both criteria.
    
    Parameters
    ----------
    
    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    perc_calc : PercentileCalculator class,
        Percentile calculator for that set of samples.     
    """
    def f(self, solution):
        perc = super(PercentileChangesCriterion, self).f(solution)
        changes = sum(solution!=self.pivot)
        
        return np.array([perc, changes])
    
    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        new_objs = self.f(new_sol)
        old_objs = self.f(old_sol)

        return all(new_objs>=old_objs)

class NonDomCriterion():
    """Class that using the changes on each feature 
    is capable of defining if a solution had to change more in all features.
    
    Parameters
    ----------
    
    pivot : numpy array,
        Sample in which we want to observe the shift and calculate the cost.
    perc_calc : PercentileCalculator class,
        Percentile calculator for that set of samples.     
    """
    def __init__(self, direc):
        self.direc=direc
    
    def greater_than(self, new_sol, old_sol):
        """Order two solutions."""
        return all(self.direc*new_sol>=self.direc*old_sol)
