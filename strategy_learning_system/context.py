#!/usr/bin/python
# Brodderick Rodriguez
# Auburn University - CSSE
# 25 Aug. 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from .feature_model import FeatureType, Feature


class Context:
	"""
	A context represents a hypothesis. the relationship between resolution model and context is 1 to 1.

	:param name: the name of this context
	:type name: str, required
	"""
	def __init__(self, name):
		"""
		Constructor method
		"""
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
		"""
		If key is an uncertainty in this resolution model

		:param key:
		:return:
		"""
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
		"""
		Add a feature to this context

		:param f:
		:return:
		"""
		if f not in self.resolution_model:
			self.resolution_model.append(f)

	@property
	def environmental_uncertainties(self):
		frm = [f for substree in self.collapsed_resolution_model() for f in substree.collapse()]
		frm = [f for f in frm if f.feature_type == FeatureType.environmental]
		return frm

	@property
	def model_uncertainties(self):
		frm = [f for substree in self.collapsed_resolution_model() for f in substree.collapse()]
		frm = [f for f in frm if f.feature_type == FeatureType.model]
		return frm
