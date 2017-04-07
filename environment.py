# -*- coding: utf-8 -*-

import numpy as np
import string


class environments:

	def __init__(self,n_states,featureMatrix) : 

		self.n_states = n_states
		self.states = list(string.lowercase[:n_states])
		self.featureMatrix = featureMatrix
		self.nfeatures = np.shape(self.featureMatrix)[1]

class mdpNstates(environments) : 

	def __init__(self,n_states,featureMatrix) : 

		environments.__init__(self,n_states,featureMatrix)

	def getNewState(self,s,a):
		position_s = self.states.index(s)
		if a == 'right' and position_s != len(self.states) - 1 : 
			new_s = self.states[position_s+1]
		elif a == 'right' and position_s == len(self.states) - 1 : 
			new_s = s
		elif a == 'left' and position_s == 0 : 
			new_s = s
		elif a == 'left' and position_s != 0 :
			new_s = self.states[position_s-1]

		return new_s

	def getReward(self,s,a):

		r = 0.
		if s == 'b' and a == 'left' : 
			r = 1.
		if s == 'a' and a == 'left' : 
			r = 1.
		
		"""
		if s == 'a' and a == 'left':
			r = 0.
		elif s == 'a' and a == 'right':
			r = 1.
		elif s == 'b' and a == 'left':
			r = 1.
		elif s == 'b' and a == 'right':
			r = 0.5
		"""

		return r

class randomWalk(environments) : 

	def __init__(self,n_states,featureMatrix) : 

		environments.__init__(self,n_states,featureMatrix)

	def isTerminal(self,s) : 
		b = False
		if s == 'TERMINAL_R':
			b = True
		elif s == 'TERMINAL_L' : 
			b= True
		return b

	def getNewState(self,s,a):
		position_s = self.states.index(s)
		if a == 'right' and position_s != len(self.states) - 1 : 
			new_s = self.states[position_s+1]
		elif a == 'right' and position_s == len(self.states) - 1 : 
			new_s = 'TERMINAL_R'
		elif a == 'left' and position_s == 0 : 
			new_s = 'TERMINAL_L'
		elif a == 'left' and position_s != 0 :
			new_s = self.states[position_s-1]

		return new_s

	def getReward(self,s,a):

		r = 0.
		if s == list(string.lowercase[:self.n_states])[-1] and a == 'right' : 
			r = 1.
		return r




