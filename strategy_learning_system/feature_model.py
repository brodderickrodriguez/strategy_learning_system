# Brodderick Rodriguez
# Auburn University - CSSE
# 29 Aug. 2019

import ema_workbench
import copy

IntegerParameter = ema_workbench.IntegerParameter
TimeSeriesOutcome = ema_workbench.TimeSeriesOutcome


class FeatureModel:
	def __init__(self):
		self.environmental_uncertainties = []
		self.model_uncertainties = []
		self.outcomes = []

	def __getitem__(self, key):
		params = dict((e.name, e) for e in self.__iter__())

		if key in params:
			return params[key]

	def __iter__(self):
		_all = (self.model_uncertainties + self.environmental_uncertainties + self.outcomes)
		return _all.__iter__()

	@property
	def environmental_uncertainties(self):
		return copy.deepcopy(self._environmental_uncertainties)

	@environmental_uncertainties.setter
	def environmental_uncertainties(self, v):
		self._environmental_uncertainties = v

	@property
	def model_uncertainties(self):
		return copy.deepcopy(self._model_uncertainties)

	@model_uncertainties.setter
	def model_uncertainties(self, v):
		self._model_uncertainties = v
	
	@property
	def outcomes(self):
		return copy.deepcopy(self._outcomes)
	
	@outcomes.setter
	def outcomes(self, v):
		self._outcomes = v

	@property
	def constraints(self):
		return self._constraints
	
	@constraints.setter
	def constraints(self, v):
		self._outcomes = v
