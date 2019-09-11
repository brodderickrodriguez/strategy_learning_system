# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import os
import pickle
from . import model_synthesizer, util
from .feature_model import FeatureModel
from .context import Context


class ModelMediator:
	def __init__(self, name=None):
		self.name = name
		self._contexts = []
		self.created_on = util.datetime_str()

	def __str__(self):
		try:
			n =  self.name
		except AttributeError:
			n = str(self.created_on)

		return 'Mediator: {}'.format(n)

	def __repr__(self):
		return self.__str__()

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, v):
		# make sure name contains no spaces
		if v is not None:
			self._name = v.replace(' ', '_')

	@property
	def model(self):
		return self._model

	@model.setter
	def model(self, v):
		# make sure that the user passed a tuple containing the model: (<dir>, <name>)
		assert isinstance(v, tuple), '{} must be a tuple: (<dir>, <name>)'.format(v)

		assert len(v) == 2, '{} must be a tuple: (<dir>, <name>)'.format(v) 

		# extract model_dir and model_name
		model_dir, model_name = v

		# if the last char in model_dir is not backslash then append it
		model_dir = util.clean_dir_path(model_dir + '/')

		# make sure that the model_dir is valid
		assert os.path.exists(model_dir), 'model dir \'{}\' does not exist'.format(model_dir)

		# make sure that the model_file exists
		assert os.path.isfile(model_dir + model_name), 'model name \'{}\' does not exist'.format(model_name)
		
		# finally save the model info
		self._model = {'dir': model_dir, 'name': model_name}

	@property
	def netlogo(self):
		return self._netlogo

	@netlogo.setter
	def netlogo(self, v):
		# make sure that the user passed a tuple containing the model: (<dir>, <version>)
		assert isinstance(v, tuple), '{} must be a tuple: (<NetLogo dir>, <version>)'.format(v)

		assert len(v) == 2, '{} must be a tuple: (<dir>, <version>)'.format(v) 

		netlogo_dir, netlogo_version = v
		
		# make sure that the model_dir is valid
		assert os.path.exists(netlogo_dir), 'NetLogo dir \'{}\' does not exist'.format(netlogo_dir)

		self._netlogo = {'dir': netlogo_dir, 'version': netlogo_version}

	@property
	def feature_model(self):
		return self._feature_model

	@feature_model.setter
	def feature_model(self, v):
		self._feature_model = v

	@property
	def save_location(self):
		return self._save_location

	@save_location.setter
	def save_location(self, save_dir):
		# make sure save_dir is valid
		assert os.path.exists(save_dir), '{} is not a valid directory'.format(save_dir)

		# save the save_location
		self._save_location = util.clean_dir_path(save_dir)

	def save(self):
		# make sure the user has specified the save_location before saving
		assert self._save_location, 'first specify the Mediator.save_location'
		
		root_dir_path = util.clean_dir_path('{}/{}'.format(self.save_location, self.name))	
		meta_data_path = root_dir_path + '/meta_data.pkl'

		# create root data directory
		if not os.path.exists(root_dir_path):
			os.mkdir(root_dir_path)

		# save the meta data
		with open(meta_data_path, 'wb') as f:
			pickle.dump(self, f)

	@staticmethod
	def load(root_dir_path):
		# make sure root_dir_path is valid
		assert os.path.exists(root_dir_path), '{} does not exist'.format(root_dir_path)

		# define the path to a Mediator's meta_data
		meta_data_path = root_dir_path + '/meta_data.pkl'

		# load meta data file
		with open(meta_data_path, 'rb') as f:
			return pickle.load(f)

	def evaluate_context(self, cxt):
		# make sure the user has specified a save location
		assert self._save_location, 'first specify the Mediator.save_location'

		# make sure cxt is of time Context
		assert isinstance(cxt, Context), '{} is not of type Context'.format(cxt)

		# make sure the cxt name is unique
		assert cxt not in self._contexts, 'a context with name {} already exists'.format(cxt.name)

		# make sure the user has specified something in the Context.resolution_model
		assert len(cxt.resolution_model) > 0, '{} has resolution_model to evaluate'.format(cxt.name)

		# make sure all model attributes in the resolution model are 
		# present in the feature model as well
		for attr in cxt.resolution_model:
			if attr not in self.feature_model:
				raise ValueError('{} is not present in the FeatureModel'.format(attr))

		# assign this context's data path
		cxt.data_path = util.clean_dir_path('{}/{}/'.format(self.save_location, self.name))	

		# add the context to this mediator
		self._contexts.append(cxt)

		# call the synthesizer to collect model data
		results = model_synthesizer.synthesize(self, cxt)

		# save the synthesized data to the context object
		cxt.raw_exploratory_results = results

	# TODO: incomplete
	def learn(self):
		pass
		# RUN XCS HERE

	# TODO: incomplete
	def explain(self):
		pass
		# MAKE HEATMAPS HERE
