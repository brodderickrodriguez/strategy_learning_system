# Brodderick Rodriguez
# Auburn University - CSSE
# 29 Aug. 2019

import copy
import ema_workbench
import enum


class FeatureType(enum.Enum):
	# red console color
	environmental = '\033[91m{}\033[0m'

	# green console color
	model = '\033[92m{}\033[0m'

	# blue console color
	outcome = '\033[94m{}\033[0m'


class FeatureConstraint(enum.Enum):
	optional = 4
	mandatory = 3
	alternative = 2
	xor = 1


class Feature:
	def __init__(self, name='', default=None):
		self._sub_features = tuple()
		self.name = name
		self.default = default
		self.feature_type: FeatureType = None
		self.constraint: FeatureConstraint = None
		self.netlogo_category = None

	def __eq__(self, other):
		def _get_name(a):
			if hasattr(a, 'name'): return a.name
			else: return a
		return _get_name(self) == _get_name(other)

	def __str__(self):
		def _subtree_rec(current, indent=0):
			display_name = current.feature_type.value.format(current.name)
			default_value = '\033[96m={}\033[0m'.format(current.default) if current.default is not None else ''
			line_details = '{}{} [{}, {}]'.format(display_name, default_value, current.feature_type, current.constraint)
			line_indent = ''.join(['│\t' for _ in range(indent)])
			line = '{}├──\t{}'.format(line_indent, line_details)
			sub_tree_lines = ''

			for sub_feature_edge in current.sub_features:
				sub_tree_lines += _subtree_rec(sub_feature_edge, indent + 1)

			return '{}\n{}'.format(line, sub_tree_lines)

		result = _subtree_rec(self)
		return '\n{}'.format(result)

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return id(self.name)

	@property
	def sub_features(self):
		return self._sub_features

	@sub_features.setter
	def sub_features(self, v):
		assert hasattr(v, '__iter__'), '{} is not iterable'.format(v)
		for vi in v: assert isinstance(vi, Feature), '{} is not of type Edge'.format(vi)
		self._sub_features = v

	def add_sub_feature(self, feature, feature_type, constraint, netlogo_category=None):
		feature.feature_type = feature_type
		feature.constraint = constraint
		feature.netlogo_category = netlogo_category
		self._sub_features = (*self._sub_features, feature)

	def collapse(self, ema_objects_only=True):
		result = []

		if not ema_objects_only or hasattr(self, 'is_ema_object'):
			s = copy.deepcopy(self)
			s.sub_features = tuple()
			result.append(s)

		for sub_feature in self.sub_features:
			result += sub_feature.collapse()

		return result

	def to_constant(self):
		return ema_workbench.Constant(str(self.name), self.default)


class IntegerParameter(Feature, ema_workbench.IntegerParameter):
	def __init__(self, *args, default=None, **kwargs):
		ema_workbench.IntegerParameter.__init__(self, *args, **kwargs)
		Feature.__init__(self, self.name, default=default)
		self.is_ema_object = True


class RealParameter(Feature, ema_workbench.RealParameter):
	def __init__(self, *args, default=None, **kwargs):
		ema_workbench.RealParameter.__init__(self, *args, **kwargs)
		Feature.__init__(self, self.name, default=default)
		self.is_ema_object = True


class CategoricalParameter(Feature, ema_workbench.CategoricalParameter):
	def __init__(self, *args, default=None, **kwargs):
		ema_workbench.CategoricalParameter.__init__(self, *args, **kwargs)
		Feature.__init__(self, self.name, default=default)

		self.is_ema_object = True
		self.should_use_default = False
		self.index_to_category = {i: v for i, v in enumerate(self._categories._data.keys())}
		self.category_to_index = {v: i for i, v in self.index_to_category.items()}


class TimeSeriesOutcome(Feature, ema_workbench.TimeSeriesOutcome):
	def __init__(self, *args, **kwargs):
		ema_workbench.TimeSeriesOutcome.__init__(self, *args, **kwargs)
		Feature.__init__(self, self.name)
		self.is_ema_object = True