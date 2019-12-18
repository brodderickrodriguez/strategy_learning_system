# Brodderick Rodriguez
# Auburn University - CSSE
# 29 Aug. 2019

import ema_workbench
import copy
import enum


class FeatureType(enum.Enum):
	# red output color
	environmental = '\033[91m{}\033[0m'

	# green output color
	model = '\033[92m{}\033[0m'

	# blue output color
	outcome = '\033[94m{}\033[0m'


class Constraint(enum.Enum):
	optional = 0
	mandatory = 1
	alternative = 2
	xor = 4


class _Edge:
	def __init__(self, feature, feature_type, constraint, netlogo_categorical_name):
		assert isinstance(feature, Feature), '{} is not of type Feature'.format(feature)
		assert isinstance(feature_type, FeatureType), '{} is not of type FeatureType'.format(feature_type)
		assert isinstance(constraint, Constraint), '{} is not of type Constraint'.format(constraint)
		self.feature = feature
		self.feature_type = feature_type
		self.constraint = constraint
		self.netlogo_categorical_name = netlogo_categorical_name


class Feature:
	def __init__(self, feature_name='', netlogo_categorical=None):
		self._sub_features = tuple()
		self.netlogo_categorical = netlogo_categorical
		self.feature_name = feature_name

	def __eq__(self, other):
		def _get_name(a):
			if hasattr(a, 'name'): return a.name
			else: return a
		return _get_name(self) == _get_name(other)

	@property
	def sub_features(self):
		return self._sub_features

	@sub_features.setter
	def sub_features(self, v):
		raise AttributeError('Feature.sub_features is not settable')

	def add_sub_feature(self, feature, feature_type, constraint, netlogo_categorical_name=None):
		sub_feature = _Edge(feature, feature_type, constraint, netlogo_categorical_name)
		self._sub_features = (*self._sub_features, sub_feature)


class IntegerParameter(ema_workbench.IntegerParameter, Feature):
	def __init__(self, *args, **kwargs):
		ema_workbench.IntegerParameter.__init__(self, *args, **kwargs)
		Feature.__init__(self)
		self.feature_name = self.name


class RealParameter(ema_workbench.RealParameter, Feature):
	def __init__(self, *args, **kwargs):
		ema_workbench.RealParameter.__init__(self, *args, **kwargs)
		Feature.__init__(self)
		self.feature_name = self.name


class CategoricalParameter(ema_workbench.CategoricalParameter, Feature):
	def __init__(self, *args, **kwargs):
		ema_workbench.CategoricalParameter.__init__(self, *args, **kwargs)
		Feature.__init__(self)
		self.feature_name = self.name
		self.index_to_category = {i: v for i, v in enumerate(self._categories._data.keys())}
		self.category_to_index = {v: i for i, v in self.index_to_category.items()}


class TimeSeriesOutcome(ema_workbench.TimeSeriesOutcome, Feature):
	def __init__(self, *args, **kwargs):
		ema_workbench.TimeSeriesOutcome.__init__(self, *args, **kwargs)
		Feature.__init__(self)
		self.feature_name = self.name


class FeatureModel:
	def __init__(self):
		self._root = Feature(feature_name='FEATURE MODEL')

	def _get_subtree(self, feature_name):
		def _subtree_rec(current):
			if current.feature_name == feature_name:
				return current
			else:
				for sub_feature_edge in current.sub_features:
					r = _subtree_rec(sub_feature_edge.feature)
					if isinstance(r, Feature):
						return r

		result = _subtree_rec(self._root)
		return result

	# refactor account for constraints
	# account for outcome, uncertainty
	def collapse(self, subtree_root_node=None, constraint=Constraint.optional):
		if subtree_root_node is None:
			subtree_root_node = self._root

		if isinstance(subtree_root_node, Feature):
			srn = subtree_root_node
		elif isinstance(subtree_root_node, _Edge):
			srn = subtree_root_node.feature

		r = [srn]

		for sub_feature in srn.sub_features:
			r += self.collapse(sub_feature)

		return r

	def subtree_str(self, subtree_root_node):
		def _subtree_rec(current, indent=0):
			if isinstance(current, _Edge):
				current_node = current.feature
				_name = current.feature_type.value.format(current_node.feature_name)
				line_details = '{} [{}, {}]'.format(_name, current.feature_type, current.constraint)
			else:
				current_node = current
				line_details = current.feature_name

			if indent == 0:
				line = line_details
			else:
				line_indent = ''.join(['│\t' for _ in range(indent)])
				line = '{}├──\t{}'.format(line_indent, line_details)

			rec_lines = ''
			for sub_feature_edge in current_node.sub_features:
				rec_lines += _subtree_rec(sub_feature_edge, indent + 1)

			return '{}\n{}'.format(line, rec_lines)

		s = _subtree_rec(subtree_root_node)
		return s

	def __str__(self):
		return self.subtree_str(self._root)

	def __iter__(self):
		return None

	def __getitem__(self, item):
		return self._get_subtree(item)

	def add_sub_feature(self, feature, feature_type, constraint, netlogo_categorical_name=None):
		self._root.add_sub_feature(feature, feature_type, constraint, netlogo_categorical_name)



# class FeatureModel:
# 	def __init__(self):
# 		pass
#
# 	def __getitem__(self, key):
# 		params = {e.name: e for e in self.__iter__()}
#
# 		if key in params:
# 			return params[key]
#
# 	def __str__(self):
# 		return str(list(self.__iter__()))
#
# 	def __iter__(self):
# 		_all = (self.model_uncertainties + self.environmental_uncertainties + self.outcomes)
# 		return _all.__iter__()
#
# 	@property
# 	def environmental_uncertainties(self):
# 		return copy.deepcopy(self._environmental_uncertainties)
#
# 	@environmental_uncertainties.setter
# 	def environmental_uncertainties(self, v):
# 		self._environmental_uncertainties = v
#
# 	@property
# 	def model_uncertainties(self):
# 		return copy.deepcopy(self._model_uncertainties)
#
# 	@model_uncertainties.setter
# 	def model_uncertainties(self, v):
# 		self._model_uncertainties = v
#
# 	@property
# 	def outcomes(self):
# 		return copy.deepcopy(self._outcomes)
#
# 	@outcomes.setter
# 	def outcomes(self, v):
# 		self._outcomes = v
#
# 	@property
# 	def constraints(self):
# 		return self._constraints
#
# 	@constraints.setter
# 	def constraints(self, v):
# 		self._outcomes = v
