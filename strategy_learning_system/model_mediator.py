# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import os
import copy
import json
from . import model_synthesizer, util
from .feature_model import FeatureModel
from .context import Context


class ModelMediator:
	def __init__(self, name=None, from_file=False):
		self.name = name
		self._contexts = []

		if not from_file:
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
		try:
			model_dir, model_name = v
		except TypeError:
			raise ValueError('must pass a tuple: (<dir>, <name>)')
			return

		# if the last char in model_dir is not backslash then append it
		model_dir = util.clean_dir(model_dir + '/')

		try:
			# make sure that the model_dir is valid
			assert os.path.exists(model_dir)

			# make sure that the model_file exists
			assert os.path.isfile(model_dir + model_name)
		except AssertionError:
			raise AttributeError('Mediator.model: {} is not a valid file'.format(model_dir + model_name))

		# finally save the model info
		self._model = {'dir': model_dir, 'name': model_name}

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
		assert os.path.exists(save_dir)

		# save the save_location
		self._save_location = util.clean_dir(save_dir)

		# save the current Mediator definition
		self.save()

	# TODO: incomplete
	def save(self):
		# make sure the user has specified the save_location before saving
		try:
			assert self._save_location
		except AttributeError:
			raise AttributeError('first specify the Mediator.save_location')
		
		root_dir_path = util.clean_dir('{}/{}'.format(self.save_location, self.name))	
		contexts_path =  root_dir_path + '/contexts'
		meta_data_path = root_dir_path + '/meta_data.json'
		feature_model_path = root_dir_path + '/feature_model.json'

		# create root data directory
		if not os.path.exists(root_dir_path):
			os.mkdir(root_dir_path)

		# create contexts dir
		if not os.path.exists(contexts_path):
			os.mkdir(contexts_path)

		# save the meta_data
		with open(meta_data_path, 'w') as f:
			meta_data_dict = copy.deepcopy(self.__dict__)

			if '_feature_model' in meta_data_dict:
				del meta_data_dict['_feature_model']

			f.write(json.dumps(meta_data_dict))

		# save feature model
		if hasattr(self, '_feature_model'):
			with open(feature_model_path, 'w') as f:
				feature_model_dict = self.feature_model.to_dict()
				f.write(json.dumps(feature_model_dict))

		# save contexts
		for ctx in self._contexts:
			ctx.save(contexts_path)

	# TODO: incomplete
	@staticmethod
	def load(root_dir_path):
		# make sure root_dir_path is valid
		assert os.path.exists(root_dir_path)

		# define the path to a Mediator's meta_data
		meta_data_path = root_dir_path + '/meta_data.json'
		feature_model_path = root_dir_path + '/feature_model.json'
		contexts_path =  root_dir_path + '/contexts'

		# create an Mediator object
		med = ModelMediator(from_file=True)

		# load and assign the meta_data 
		with open(meta_data_path, 'r') as f:
			med.__dict__ = json.load(f)

		# load feature model
		med.feature_model = FeatureModel.from_file(feature_model_path)

		# load contexts
		for cxt_name in os.listdir(contexts_path):
			cxt_path = '{}/{}'.format(contexts_path, cxt_name)
			cxt = Context.load(cxt_path)
			med._contexts.append(cxt)

		return med

	def evaluate_context(self, cxt):
		# make sure cxt is of time Context
		if not isinstance(cxt, Context):
			raise ValueError('{} is not of type Context'.format(cxt))

		# name sure the cxt name is unique
		if True in [cxt.name == o.name for o in self._contexts]:
			raise NameError('a context with then name {} already exists'.format(cxt.name))

		# make sure all model attributes in the resolution model are 
		# present in the feature model as well
		for attr in cxt.resolution_model:
			if attr not in self.feature_model:
				raise ValueError('{} is not present in the FeatureModel'.format(attr))

		# save this context 
		contexts_path = util.clean_dir('{}/{}/contexts/'.format(self.save_location, self.name))	
		cxt.save(contexts_path)

		# add the context to this mediator
		self._contexts.append(cxt)

		# call the synthesizer to collect model data
		results = model_synthesizer.synthesize(self, cxt)

		# TODO: this can maybe be moved to 
		# save the synthesized data to the context object
		cxt.synthesized_data = results
	

	# TODO: incomplete
	def learn(self):
		pass
		# RUN XCS HERE

	# TODO: incomplete
	def explain(self):
		pass
		# MAKE HEATMAPS HERE
