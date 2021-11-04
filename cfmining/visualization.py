    # -*- coding: utf-8 -*-
"""
@author:  Marcos M. Raimundo
@email:   marcosmrai@gmail.com
@license: BSD 3-clause.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from cycler import cycler
import seaborn as sns
import warnings

def buildTable(enumerator, pivot, criteria, names=None,
                  include_original=True, include_cost=True):
    """Creates a pandas dataframe showing all actionable recourses.
    
    Parameters
    ----------
    
    enumerator : MAPOCAM class,
        Optimizer with solutions parameters.
    pivot : numpy type array,
        Sample that had an undesired prediction that we want to find recourses
        that change the outcome.
    criteria : criteria type class.
        Class that evaluate the cost of a recourse.
    names : list os str.
        Names of the features.
    include_original : bool.
        Bolean that indicates if pivot is included or not in the first colmuns.
    include_cost : bool.
        Bolean that indicates if cost is included or not in the last row.
    """
    if names is None:
        names = np.arange(len(pivot))

    if include_original:    
        overview = pd.DataFrame(pivot[:,np.newaxis], index=names, columns=["Orig"])
        if include_cost:
            overview.loc['Cost'] = np.nan
    else:
        overview = pd.DataFrame(index=names)
    
    eita = np.array([criteria.f(act)
                          for act in enumerator.solutions])
    
    sort_idx = np.argsort([criteria.f(act)
                          for act in enumerator.solutions])
    
    for idx, act_idx in enumerate(sort_idx):
        act = enumerator.solutions[act_idx].copy().astype('float')
        act[act==pivot] = np.nan
        if include_cost:
            overview['C'+str(idx+1)] = np.concatenate([act, [criteria.f(act)]])
        else:
            overview['C'+str(idx+1)] = act
    
    return overview

class FeatureImportance():
    """Creates an index of how much a features would need to change to solely
    create an actionable recourse.
   
    Parameters
    ----------
    
    enumerator : MAPOCAM class,
        Optimizer with solutions parameters.
    pivot : numpy type array,
        Sample that had an undesired prediction that we want to find recourses
        that change the outcome.
    action_set : ActionSet type class,
        Contains the discretization of the features to find the recourses.
    """
    def __init__(self, enumerator, pivot, action_set):
        self.enum = enumerator
        self.pivot = pivot
        self.action_set = action_set
    
    def calc(self):
        cost_table = np.zeros((self.pivot.size, len(self.enum.solutions)))
        for nsol, sol in enumerate(self.enum.solutions):
            changes = sum(sol!=self.pivot)
            for idx, feat in enumerate(self.enum.names):
                #(self.enum.feas_grid[feat][-1]-self.enum.feas_grid[feat][0])
                #div by zero
                if (self.enum.feas_grid[feat][-1]-self.enum.feas_grid[feat][0])!=0:
                    cost_table[idx, nsol] = (sol[idx]-self.enum.feas_grid[feat][0])/(self.enum.feas_grid[feat][-1]-self.enum.feas_grid[feat][0])*changes
                else:
                    cost_table[idx, nsol]=np.nan
        cost_table[np.where(cost_table==0)]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            change_level = pd.Series(np.nanmean(cost_table,axis=1), index=self.enum.names)
        self.change_level = change_level
        return self.change_level
    
        
    def show(self, path=None):
        try:
            self.change_level
        except:
            self.calc()
        plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('deep', 20))
        plt.rcParams["figure.figsize"] = [3.0,4.0]
        colors = ['#d9d9d9' if v else '#1f78b4' for v in self.change_level.isna()]        
        self.change_level[self.change_level.isna()] = self.change_level.max()
        ax = self.change_level.plot.barh(legend=False, color=colors);
        ax.tick_params(labelleft=False) 
        if path is not None:
            ax.figure.savefig(path, bbox_inches='tight')
        return ax

class PlotCounterfactuals():
    """How all options of actionable recourses.
    """ 
    def __init__(self, actions, individual=None, apx=2, exclude_nanrows=True, height=None):
        self.boxes = []
        if exclude_nanrows:
            changes = (~actions.isna()).sum(axis=1)
            new_order = np.argsort(-changes[changes!=0].values)
            self.actions = actions[changes!=0]
            self.actions = self.actions.iloc[new_order]
            if individual is not None:
                self.actions.insert(0, 'Original', individual[changes!=0][new_order], True)
        else:
            self.actions = actions
            if individual is not None:
                self.actions.insert(0, 'Original', individual, True)
        self.H, self.W = self.actions.shape
        if height is None:
            self.height = [(self.H-i-1,a) for i,a in enumerate(self.actions.index)]
        else:
            self.height = height
        self.apx = apx
        
    def value(self, v):
        if float(v).is_integer() or float(v)>5:
            return str(int(v))
        elif v<1:
            return str(round(v,self.apx+1))[1:]
        else:
            return str(round(v,self.apx))
        
    def show(self, path=None, hl_column=[], hl_row=[], bbox_to_anchor=(-0.01,0)):
        plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('pastel', 20))
        plt.rcParams["figure.figsize"] = np.array(self.actions.shape)[::-1]*0.5
        self.fig, self.ax = plt.subplots()
        for i in range(1, self.W):
            self.ax.plot([i, i],[-0.5, self.H-0.5], 'k--', linewidth=0.1, alpha=0.2)

        for i,feat in self.height:
            x = [value if ~np.isnan(ind) else np.nan for value, ind in zip(np.arange(self.W), self.actions.loc[feat].values)]
            self.ax.plot(x, i*np.ones(self.W), 's', label=feat, markersize=20)
            for xi in x:
                if ~np.isnan(xi):
                    self.ax.text(xi, i, self.value(self.actions.loc[feat].values[xi]), ha="center", va="center", fontsize=8.5)
        self.ax.set_ylim([-0.5, self.H-0.5])
        self.ax.set_xlim([-0.5, self.W-0.5])
        l = mlines.Line2D([0.5, 0.5], [-0.5, self.H-0.5],  color='black', linewidth='1')
        self.ax.add_line(l)
        rect = Rectangle((-0.5,-0.5),1,self.H,linewidth=0, edgecolor='k',facecolor='gainsboro')
        self.ax.add_patch(rect)
        for hl in hl_column:
            rect = Rectangle((-0.46+hl,-0.46),0.9,self.H-0.1, linewidth=1, linestyle='--', edgecolor='k', alpha=1, facecolor='none')
            self.ax.add_patch(rect)
            
        for hl in hl_row:
            rect = Rectangle((-0.46,-0.46+hl),self.W-0.1, 0.9, linewidth=1, linestyle='--', edgecolor='k', alpha=1, facecolor='none')
            self.ax.add_patch(rect)
        
        self.ax.legend(title = 'Feature names', loc='lower right', handletextpad=1,  borderaxespad=0.0 , bbox_to_anchor=bbox_to_anchor, fancybox=True, labelspacing=1.68, borderpad=1)
        self.ax.set_aspect('equal')
        
        plt.xticks(np.arange(self.W), ['Orig']+['C'+str(i) for i in range(1,self.W)])
        plt.yticks([], [])
        self.ax.xaxis.set_ticks_position('top') 
        if path is not None:
            plt.savefig(path, bbox_inches='tight',pad_inches = 0.01)
        plt.show()
    
