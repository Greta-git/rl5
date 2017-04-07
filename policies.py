# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rv_discrete

class policies :
    
    def __init__(self,actions,proba_mu,proba_pi) : 
        self.actions = actions
        self.behaviorProba = proba_mu
        self.targetProba = proba_pi
        
    def behaviorPolicy(self) :
        x = [1,2]
        px = self.behaviorProba
        a = int(rv_discrete(values=(x,px)).rvs(size=1))
        return self.actions[a-1]              

    def targetPolicy(self) : 
        x = [1,2]
        px = self.targetProba
        a = int(rv_discrete(values=(x,px)).rvs(size=1))
        return self.actions[a-1] 
    
    def getRho(self,a) : 
    	# in a more general case, getRho depends on the state and we instead define 2 matrices pi and mu : |A| x |S|
        position_a = self.actions.index(a)
        rho = self.targetProba[position_a] / self.behaviorProba[position_a]
        return rho
