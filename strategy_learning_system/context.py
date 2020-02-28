# Brodderick Rodriguez
# Auburn University - CSSE
# 25 Aug. 2019

import numpy as np
from .feature_model import FeatureType, Feature


class Context:
	def __init__(self, name):
		self.name = name
		self.resolution_model = []
		self.bins = np.linspace(0.0, 1.0, 3)

		self.num_experiments = 0
		self.num_replications = 0
		self.max_run_length = 0
		self.num_processes = 1

		self.exploratory_data = None
		self.learned_data = None

	def __str__(self):
		return 'Context: %s' % self.name

	def __repr__(self):
		return self.__str__()

	def __eq__(self, o):
		return self.name == o.name

	def __getitem__(self, key):
		def _look(l):
			for f in l:
				if f.name == key:
					return f
			return None

		features = self.collapsed_resolution_model()
		frm = _look(features)
		return frm

	def collapsed_resolution_model(self):
		features = []

		for subtree in self.resolution_model:
			if isinstance(subtree, Feature):
				sub_features = [feat for feat in subtree.collapse()]
				features += sub_features

		return features

	@staticmethod
	def reward_function(outcome_keys, outcomes):
		raise NotImplementedError

	def add_feature(self, f):
		if f not in self.resolution_model:
			self.resolution_model.append(f)

	def environmental_uncertainties(self):
		frm = [f for substree in self.collapsed_resolution_model() for f in substree.collapse()]
		frm = [f for f in frm if f.feature_type == FeatureType.environmental]
		return frm

	def model_uncertainties(self):
		frm = [f for substree in self.collapsed_resolution_model() for f in substree.collapse()]
		frm = [f for f in frm if f.feature_type == FeatureType.model]
		return frm
