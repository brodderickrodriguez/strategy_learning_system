# Brodderick Rodriguez
# Auburn University - CSSE
# 25 Aug. 2019

import os
from . import util
import numpy as np
import pandas as pd
import pickle
from .feature_model import FeatureType, IntegerParameter


class Rule:
	def __init__(self, classifier=None):
		self.model_uncertainties = []
		self.environmental_uncertainties = []
		self.outcome = 0
		self.confidence = 0
		self.experience = 0
		self.classifier = classifier

	def __str__(self):
		s = '\nIF:\t  {0}\nWHEN: {1}\nTHEN: {2:0.3f}\n(conf.: {3:0.3f}, exp.: {4})\n'
		f_i = '\n\t{a} <= {fname} <= {b}'
		s_m, s_e = '', ''

		for f, r, _ in self.model_uncertainties:
			s_m += f_i.format(a=r[0], b=r[1], fname=f.name)

		for f, r, _ in self.environmental_uncertainties:
			s_e += f_i.format(a=r[0], b=r[1], fname=f.name)

		return s.format(s_m, s_e, self.outcome, self.confidence, self.experience)

	def __repr__(self):
		return str(self)

	def __lt__(self, other):
		sv = self.outcome * self.confidence
		ov = other.outcome * other.confidence
		return sv < ov

	def __eq__(self, other):
		return self.model_uncertainties == other.model_uncertainties and \
			   self.environmental_uncertainties == other.environmental_uncertainties

	# def __hash__(self):
	# 	return self.model_uncertainties, self.environmental_uncertainties

	@staticmethod
	def from_xcsr_classifier(classifier, env_uncertainty, mod_uncertainty, bins):
		rule = Rule(classifier)
		rule.outcome = classifier.predicted_payoff
		rule.confidence = 1 - classifier.epsilon
		rule.experience = classifier.experience

		def _get_true_value(f_range, uncertainty):
			r = lambda p: p * (uncertainty.upper_bound - uncertainty.lower_bound) + uncertainty.lower_bound
			a, b = r(f_range[0]), r(f_range[1])

			if isinstance(eu, IntegerParameter):
				a, b = int(a), int(b)

			return a, b

		def _get_bin_range(a):
			a = int(a)
			l = a - 1 if a > 0 else 0
			r = a if a != 0 else 1
			return bins[l], bins[r]

		for pred, eu in zip(classifier.predicate, env_uncertainty):
			p_true = _get_true_value(pred, eu)
			rule.environmental_uncertainties.append((eu, p_true, pred))

		for act, mu in zip(classifier.action, mod_uncertainty):
			act_range = _get_bin_range(act)
			act_true = _get_true_value(act_range, mu)
			rule.model_uncertainties.append((mu, act_true, act_range))

		return rule


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
		features = [feat for subtree in self.resolution_model for feat in subtree.collapse()]
		return features

	@staticmethod
	def reward_function(outcome_keys, outcomes):
		raise NotImplementedError

	def add_feature(self, f):
		if f not in self.resolution_model:
			self.resolution_model.append(f)

	@property
	def raw_exploratory_results(self):
		# define a path to the raw exploratory results for this context
		path = Context.RAW_EXPL_RES_PATH.format(self.data_path, self.name)

		# assert that the path exists
		# i.e. the exploration process has be done
		assert os.path.exists(path), 'the exploration data cannot be found for {}'.format(self.name)

		# use pickle to load and convert the byte stream to exploration data
		with open(path, 'rb') as f:
			return pickle.load(f)

	@raw_exploratory_results.setter
	def raw_exploratory_results(self, v):
		# define a path to the raw exploratory results for this context
		path = Context.RAW_EXPL_RES_PATH.format(self.data_path, self.name)

		# save the raw exploratory results using pickle byte steam
		with open(path, 'wb') as f:
			pickle.dump(v, f)

		# call the process function to process the exploration data
		self.process_exploratory_results(v)
	
	def process_exploratory_results(self, results):
		# reshape and normalize the exploratory results data
		results = util.process_ema_results(context=self, results=results)

		# separate experiments and outcomes
		experiments = results['experiments']

		# call the user-designed reward function to convert raw outcomes to reward
		outcomes = self.reward_function(*results['outcomes'])

		# create a new panda data frame to contain all the data
		data = experiments.copy()

		# insert the reward to the last column in the new data frame
		data.insert(len(data.columns), 'rho', outcomes)

		# set the processed results to a variable
		self.processed_exploratory_results = data

	@property
	def raw_learned_results(self):
		# define a path to the raw learned results for this context
		path = Context.RAW_LEAR_RES_PATH.format(self.data_path[:-1], self.name)

		# assert that the path exists
		# i.e. the learning process has be done
		assert os.path.exists(path), 'the learned data cannot be found for {}'.format(self.name)

		# use pickle to load and convert the byte stream to exploration data
		with open(path, 'rb') as f:
			return pickle.load(f)

	@raw_learned_results.setter
	def raw_learned_results(self, v):
		# define a path to the raw learned results for this context
		path = Context.RAW_LEAR_RES_PATH.format(self.data_path, self.name)

		# save the raw learned results
		with open(path, 'wb') as f:
			pickle.dump(v, f)

		# call the process function to process the learned data
		self.process_learned_results(v)

	def process_learned_results(self, results):
		env_uncertainty = self.environmental_uncertainties()
		mod_uncertainty = self.model_uncertainties()
		classifiers = [cl for rep in results for cl in rep]
		rules = []

		for cl in classifiers:
			if cl.experience == 0:
				continue
			else:
				r = Rule.from_xcsr_classifier(cl, env_uncertainty, mod_uncertainty, self.bins)
				rules.append(r)

		self.processed_learned_data = sorted(rules, reverse=True)

	def environmental_uncertainties(self):
		frm = [f for substree in self.resolution_model for f in substree.collapse()]
		frm = [f for f in frm if f.feature_type == FeatureType.environmental]
		return frm

	def model_uncertainties(self):
		frm = [f for substree in self.resolution_model for f in substree.collapse()]
		frm = [f for f in frm if f.feature_type == FeatureType.model]
		return frm

	def update_bins(self, new_bins):
		self.bins = new_bins
		re = self.raw_exploratory_results
		self.process_exploratory_results(re)
