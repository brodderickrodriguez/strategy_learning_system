# Brodderick Rodriguez
# Auburn University - CSSE
# 25 Aug. 2019

from . import util
import numpy as np
from .feature_model import FeatureType, Feature


class Context:
	RAW_EXPL_RES_PATH = '{}/{}_raw_exploratory_results.pkl'
	RAW_LEAR_RES_PATH = '{}/{}_raw_learned_results.pkl'

	def __init__(self, name=None):
		dts = util.datetime_str()

		self.name = name if name is not None else dts
		self.created_on = dts
		self.resolution_model = []
		self.all_parameters = []
		self.data_path = None
		self.bins = np.linspace(0.0, 1.0, 3)

		self.num_experiments = 0
		self.num_replications = 0
		self.max_run_length = 0
		self.num_processes = 1
		self.tasks_per_subchild = 4

		self.processed_exploratory_results = None
		self.processed_learned_data = None

	def __str__(self):
		return 'Context: {}'.format(self.name)

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

		if frm is not None:
			return frm

		apl = _look(self.all_parameters)
		return apl

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
