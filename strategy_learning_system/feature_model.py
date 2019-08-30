# Brodderick Rodriguez
# Auburn University - CSSE
# 29 Aug. 2019

import ema_workbench
import copy
import json

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

	def to_dict(self):
		d = copy.deepcopy(self.__dict__)
		for key, val in d.items():
			d[key] = [o.__dict__ for o in val]
		return d

	@staticmethod
	def from_file(f):
		with open(f, 'r') as f:
			fdr = json.load(f)

		fm = FeatureModel()

		fm.all_outcomes = [TimeSeriesOutcome(o['name']) 
							for o in fdr['_outcomes']]

		fm.environmental_uncertainties = [IntegerParameter(o['name'], o['lower_bound'], o['upper_bound']) 
											for o in fdr['_environmental_uncertainties']]

		fm.model_uncertainties = [IntegerParameter(o['name'], o['lower_bound'], o['upper_bound']) 
											for o in fdr['_model_uncertainties']]

		return fm
