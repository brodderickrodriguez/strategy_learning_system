# Brodderick Rodriguez
# Auburn University - CSSE
# 29 Aug. 2019

import ema_workbench
import copy


def _compare_params(a, b):
	def _get_name(c):
		if hasattr(c, 'name'):
			return c.name
		else:
			return c

	return _get_name(a) == _get_name(b)


class IntegerParameter(ema_workbench.IntegerParameter):
	def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
	def __eq__(self, other): return _compare_params(self, other)


class RealParameter(ema_workbench.RealParameter):
	def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
	def __eq__(self, other): return _compare_params(self, other)


class CategoricalParameter(ema_workbench.CategoricalParameter):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.index_to_category = {i: v for i, v in enumerate(self._categories._data.keys())}
		self.category_to_index = {v: i for i, v in self.index_to_category.items()}

	def __eq__(self, other):
		return _compare_params(self, other)


class TimeSeriesOutcome(ema_workbench.TimeSeriesOutcome):
	def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
	def __eq__(self, other): return _compare_params(self, other)


class FeatureModel:
	def __init__(self):
		pass

	def __getitem__(self, key):
		params = {e.name: e for e in self.__iter__()}

		if key in params:
			return params[key]

	def __str__(self):
		return str(list(self.__iter__()))

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
