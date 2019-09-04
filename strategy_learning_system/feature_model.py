# Brodderick Rodriguez
# Auburn University - CSSE
# 29 Aug. 2019

import ema_workbench

IntegerParameter = ema_workbench.IntegerParameter
TimeSeriesOutcome = ema_workbench.TimeSeriesOutcome


class FeatureModel:
	def __init__(self):
		pass

	@property
	def environmental_uncertainties(self):
		return self._environmental_uncertainties

	@environmental_uncertainties.setter
	def environmental_uncertainties(self, v):
		self._environmental_uncertainties = v

	@property
	def model_uncertainties(self):
		return self._model_uncertainties

	@model_uncertainties.setter
	def model_uncertainties(self, v):
		self._model_uncertainties = v
	
	@property
	def outcomes(self):
		return self._outcomes
	
	@outcomes.setter
	def outcomes(self, v):
		self._outcomes = v

	@property
	def constraints(self):
		return self._constraints
	
	@constraints.setter
	def constraints(self, v):
		self._outcomes = v

	def __getitem__(self, key):
		print('getting ', key)

	def __iter__(self):
		_all = (self.model_uncertainties + self.environmental_uncertainties + self.outcomes)
		return _all.__iter__()
