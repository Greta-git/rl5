# -*- coding: utf-8 -*-

import numpy as np

class ETD0 : 
    
    def __init__(self,alpha,beta,gamma,env) : 
    	self.beta = beta
    	self.alpha = alpha
    	self.gamma = gamma
    	self.featureMatrix = env.featureMatrix

    def getUpdatesETD0(self,s,new_s,terminalState,F_t_m1,theta_t,rho_t,rho_t_m1,r):
    	# states has to be number in order to be identified in the feature matrix

    	fi_s = self.featureMatrix[s][:,None]

    	F_t = self.beta * rho_t_m1 * F_t_m1 + 1
        
        if terminalState == False : 
            fi_s_p1 = self.featureMatrix[new_s][:,None]
        elif terminalState == True :
            fi_s_p1 = np.zeros((np.shape(self.featureMatrix)[1],1))

        theta_t_p1 = theta_t + self.alpha * F_t * rho_t * (r + self.gamma * np.dot(theta_t.T,fi_s_p1) - np.dot(theta_t.T,fi_s))*fi_s
        
        return theta_t_p1,F_t


class TD0 : 
    
    def __init__(self,alpha,gamma,env) : 
        self.alpha = alpha
        self.gamma = gamma
        self.featureMatrix = env.featureMatrix

    def getUpdatesTD0(self,s,new_s,terminalState,theta_t,rho_t,r):
        # states has to be number in order to be identified in the feature matrix

        fi_s = self.featureMatrix[s][:,None]
        
        if terminalState == False : 
            fi_s_p1 = self.featureMatrix[new_s][:,None]
        elif terminalState == True :
            fi_s_p1 = np.zeros((np.shape(self.featureMatrix)[1],1))

        theta_t_p1 = theta_t + self.alpha * rho_t * (r + self.gamma * np.dot(theta_t.T,fi_s_p1) - np.dot(theta_t.T,fi_s))*fi_s
        
        return theta_t_p1

