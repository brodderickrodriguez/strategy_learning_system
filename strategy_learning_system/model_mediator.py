# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import os
import pickle
from . import explore, learn, explain
from .context import Context
import warnings


class ModelMediator:
	def __init__(self, name):
		self.name = name
		self.feature_model = None
		self._save_location = None
		self._contexts = {}
		self._model_info = {}
		self._netlogo_info = {}

	def __str__(self):
		return 'Mediator: %s' % self.name

	def __repr__(self):
		return str(self)

	def __getitem__(self, key):
		if key not in self._contexts:
			raise RuntimeError('Context %s not found' % key)

		return self._contexts[key]

	def __setitem__(self, key):
		warnings.warn('this operation is not supported')

	@property
	def model(self):
		return self._model_info

	@model.setter
	def model(self, val):
		# make sure that the user passed a tuple containing the model: (<dir>, <name>)
		assert isinstance(val, tuple), '{} must be a tuple: (<dir>, <name>)'.format(val)

		assert len(val) == 2, '{} must be a tuple: (<dir>, <name>)'.format(val)

		# extract model_dir and model_name
		model_dir, model_name = val

		# make sure that the model_dir is valid
		assert os.path.exists(model_dir), 'model dir \'{}\' does not exist'.format(model_dir)

		# make sure that the model_file exists
		assert os.path.isfile(model_dir + model_name), 'model name \'{}\' does not exist'.format(model_name)
		
		# finally save the model info
		self._model_info = {'dir': model_dir, 'name': model_name}

	@property
	def netlogo(self):
		return self._netlogo_info

	@netlogo.setter
	def netlogo(self, v):
		# make sure that the user passed a tuple containing the model: (<dir>, <version>)
		assert isinstance(v, tuple), '{} must be a tuple: (<NetLogo dir>, <version>)'.format(v)

		assert len(v) == 2, '{} must be a tuple: (<dir>, <version>)'.format(v) 

		netlogo_dir, netlogo_version = v
		
		# make sure that the model_dir is valid
		assert os.path.exists(netlogo_dir), 'NetLogo dir \'{}\' does not exist'.format(netlogo_dir)

		self._netlogo_info = {'dir': netlogo_dir, 'version': netlogo_version}

	@property
	def features(self):
		return self.feature_model

	@property
	def save_location(self):
		return self._save_location

	@save_location.setter
	def save_location(self, save_dir):
		# make sure save_dir is valid
		assert os.path.exists(save_dir), '%s is not a valid directory' % save_dir

		self._save_location = save_dir

	def remove_context(self, cxt_name):
		if cxt_name in self._contexts:
			del self._contexts[cxt_name]
		else:
			warnings.warn('Context %s not found' % cxt_name)

	def save(self):
		# make sure the user has specified the save_location before saving
		assert self._save_location, 'first specify Mediator.save_location'

		root_dir_file_path = '%s/%s.pkl' % (self.save_location, self.name)

		# save the meta data
		with open(root_dir_file_path, 'wb') as f:
			pickle.dump(self, f)

	@staticmethod
	def load(root_dir_path):
		root_dir_file_path = '%s.pkl' % root_dir_path

		# make sure root_dir_path is valid
		assert os.path.exists(root_dir_file_path), '%s does not exist' % root_dir_path

		# load meta data file
		with open(root_dir_file_path, 'rb') as f:
			return pickle.load(f)

	def explore(self, cxt):
		# make sure cxt is of time Context
		assert isinstance(cxt, Context), '{} is not of type Context'.format(cxt)

		# make sure the cxt name is unique
		assert cxt.name not in self._contexts, 'a context with name {} already exists'.format(cxt.name)

		# make sure the feature model has been specified
		assert self.feature_model is not None, 'a feature model has not been specified'

		# make sure the user has specified something in the Context.resolution_model
		assert len(cxt.resolution_model) > 0, '{} has resolution_model to evaluate'.format(cxt.name)

		if cxt.exploratory_data is not None:
			warnings.warn('Overriding explored data for Context %s' % cxt.name)

		# add the context to this mediator
		self._contexts[cxt.name] = cxt

		# call the synthesizer to collect model data
		results = explore.explore(self, cxt)

		# save the synthesized data to the context object
		cxt.exploratory_data = results

		# return the data for fast access
		return results

	def learn(self, cxt, algorithm='mlp_hac'):
		# make sure cxt is of time Context
		assert isinstance(cxt, Context), '{} is not of type Context'

		# make sure the context has been explored
		assert cxt.exploratory_data is not None, 'this context has not been explored'

		if cxt.learned_data is not None:
			warnings.warn('Overriding learned data for Context %s' % cxt.name)

		# call the learning module and collect the learned rules
		rules = learn.learn(self, cxt, algorithm)

		# save the learned rules to the context object
		cxt.learned_data = rules

		return rules

	@staticmethod
	def explain(cxt):
		# make sure cxt is of time Context
		assert isinstance(cxt, Context), '{} is not of type Context'

		if cxt.exploratory_data is not None:
			explain.plot_explored(cxt)
		else:
			warnings.warn('%s has not been explored' % cxt.name)

		if cxt.learned_data is not None:
			explain.plot_learned(cxt)
		else:
			warnings.warn('%s has no learned data' % cxt.name)
