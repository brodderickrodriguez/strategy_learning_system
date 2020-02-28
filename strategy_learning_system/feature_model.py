# Brodderick Rodriguez
# Auburn University - CSSE
# 29 Aug. 2019

import copy
from .feature import Feature, FeatureType, FeatureConstraint


class FeatureModel:
	def __init__(self):
		self._root = Feature('feature_model')
		self._root.feature_type = FeatureType.model
		self._root.constraint = FeatureConstraint.mandatory

	def __str__(self):
		return str(self._root)

	def __iter__(self):
		return self._root.collapse().__iter__()

	def _recurse_get_item(self, item, include_children):
		def _subtree_rec(current):
			if current.name == item:
				if not include_children:
					current_ = copy.deepcopy(current)
					current_._sub_features = tuple()
					return current_
				return current
			else:
				for sub_feature_edge in current.sub_features:
					r = _subtree_rec(sub_feature_edge)
					if r is not None:
						return r

		features_ = _subtree_rec(self._root)
		return features_

	def get_item(self, item, include_children=True):
		features_ = self._recurse_get_item(item, include_children=include_children)
		return features_

	def __getitem__(self, item):
		features_ = self._recurse_get_item(item, include_children=True)
		return features_

	def __contains__(self, item):
		for p in list(self.__iter__()):
			if p == item.name:
				return True

		return False

	def collapse(self, subtree=None):
		if subtree is None:
			subtree = self._root
		return subtree.collapse()

	def add_sub_feature(self, feature, feature_type, constraint, netlogo_categorical_name=None):
		self._root.add_sub_feature(feature, feature_type, constraint, netlogo_categorical_name)

	def _get_uncertainties(self, feature_type):
		def _subtree_rec(current):
			env_uncertainties = []
			if current.feature_type == feature_type:
				c = copy.deepcopy(current)
				c.sub_features = tuple()
				env_uncertainties.append(c)

			for sub_feature in current.sub_features:
				env_uncertainties += _subtree_rec(sub_feature)

			return env_uncertainties

		result = _subtree_rec(self._root)
		return result

	def model_uncertainties(self):
		result = self._get_uncertainties(feature_type=FeatureType.model)
		return result

	def environmental_uncertainties(self):
		result = self._get_uncertainties(feature_type=FeatureType.environmental)
		return result
