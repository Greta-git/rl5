# -*- coding: utf-8 -*-

import numpy as np
import environment
import policies
import algorithm


def computeMSE(value1,true_values,D) : 

	return ((value1 - true_values).T).dot(D).dot(value1 - true_values)

def means_dict(dictionary) : 

	nb_runs = len(dictionary.keys())
	length = len(dictionary[dictionary.keys()[0]])

	table_all_runs = np.array((nb_runs,length))
	for i in range(nb_runs) : 
		table_all_runs[i] = np.array(dictionary[i])

	av1 = np.mean(table_all_runs,axis = 0)

	return av1

def projectionMatrix(featureMatrix,D) : 

	b = (featureMatrix.T).dot(D)
	c = (featureMatrix.T).dot(D).dot(featureMatrix)
	try:
		np.linalg.inv(c)
		c_inv = np.linalg.inv(c)
	except:
		c_inv = np.linalg.pinv(c)
	projMatrix = featureMatrix.dot(c_inv).dot(b)
	return projMatrix

def TV(theta,featureMatrix,transitionMatrix,R,gamma) :

	TV = R + gamma * transitionMatrix.dot(featureMatrix).dot(theta)

	return TV 

def getVectorF(d_mu,transitionMatrix,n_states,beta):
	c = np.eye(n_states) - beta * transitionMatrix
	c2 = np.linalg.inv(c)
	d = np.array(d_mu)[:,None]
	f = (d.T).dot(c2)
	return f.T


class exp : 

	def __init__(self,environment,policies,algo_class) : 
		self.env = environment
		self.pol = policies
		self.algo = algo_class
		self.featureMatrix = self.env.featureMatrix

	def one_run_generalUpdate(self,s0,maxSteps,episodic,algo):
		# update the theta at each step within an episode but plot it for each episode

		thetas = []

		# initiate the values
		F_t_m1 = 1.
		rho_t_m1 = 1.
		theta_t = np.zeros((self.env.nfeatures,1))

		# start running
		count_step = 0
		s = s0
		thetas.append(theta_t)

		while count_step < maxSteps : 

			# take action, observe reward and new_state
			a = self.pol.behaviorPolicy()
			r = self.env.getReward(s,a)
			new_s = self.env.getNewState(s,a)
			#print 'state: ',s
			#print 'action: ',a
			#print 'reward: ',r
			
			terminalSt = False
			try:
				self.env.isTerminal(new_s)
				if  self.env.isTerminal(new_s) == True:
					terminalSt = True
					position = 12 # wrong position, will be adjusted in the update function
			except:
				pass

			if terminalSt == False : 
				position = self.env.states.index(new_s)
		
			# get rho with the current state and the chosen action
			rho_t = self.pol.getRho(a)

			# get the updates
			if algo == 'ETD0' : 
				theta_t_p1,F_t = self.algo.getUpdatesETD0(self.env.states.index(s),position,terminalSt,F_t_m1,theta_t,rho_t,rho_t_m1,r)
			if algo == 'TD' : 
				theta_t_p1 = self.algo.getUpdatesTD0(self.env.states.index(s),position,terminalSt,theta_t,rho_t,r)
				
			theta_t = theta_t_p1

			# memory the previous value of rho
			rho_t_m1 = np.copy(rho_t)

			# go to new state and update count of steps. If episodic: max number of episodes. get to next episode when s == terminal state
			try:
				self.env.isTerminal(new_s)
				if  self.env.isTerminal(new_s) == True:
					count_step += 1
					s = s0
					thetas.append(theta_t)	

				elif self.env.isTerminal(new_s) == False:
					s = new_s
			except:
				pass

			if episodic == False : 
				count_step += 1
				s = new_s
				thetas.append(theta_t)	

		return thetas	

	def multiple_runs(self,s0,maxSteps,episodic,algo,maxRuns):

		count_run = 0
		runs_thetas = dict()
		
		while count_run < maxRuns : 
			
			print '\t run %s being done'%count_run		
			
			thetas = self.one_run_generalUpdate(s0,maxSteps,episodic,algo)
			
			runs_thetas[count_run] = thetas

			count_run += 1
		
		return runs_thetas

def getMSE_allP_oneBeta(actions,proba_pi,alpha,gamma,environment,beta,s0,maxSteps,episodic,algo,nb_runs) : 
	if algo == 'ETD0' : 
		algorithm_etd8_rw5 = algorithm.ETD0(alpha,beta,gamma,environment)
	if algo == 'TD' : 
		algorithm_etd8_rw5 = algorithm.TD0(alpha,gamma,environment)
	valeurs_p = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	thetas_p_8 = dict()
	for p in valeurs_p :
	    print 'p ',p,' being done'
	    proba_mu= [p,1-p]
	    policies_rw5 = policies.policies(actions,proba_mu,proba_pi)
	    experiments_rw5 = exp(environment,policies_rw5,algorithm_etd8_rw5)
	    
	    runs_thetas = experiments_rw5.multiple_runs(s0,maxSteps,episodic,algo,nb_runs)
	    thetas_p_8[p] = runs_thetas

	return thetas_p_8
