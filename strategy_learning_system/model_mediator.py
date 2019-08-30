# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import os
import copy
import json
import pickle
import datetime
from . import model_synthesizer, util
from .feature_model import FeatureModel


class ModelMediator:
	def __init__(self, name=None, from_file=False):
		self.name = name

		if not from_file:
			self.created_on = str(datetime.datetime.now())

	def __str__(self):
		try:
			return self.name
		except AttributeError:
			return str(self.created_on)

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

		# save contexts here

	# TODO: incomplete
	@staticmethod
	def load(root_dir_path):
		# make sure root_dir_path is valid
		assert os.path.exists(root_dir_path)

		# define the path to a Mediator's meta_data
		meta_data_path = root_dir_path + '/meta_data.json'
		feature_model_path = root_dir_path + '/feature_model.json'

		# create an Mediator object
		med = ModelMediator(from_file=True)

		# load and assign the meta_data 
		with open(meta_data_path, 'r') as f:
			med.__dict__ = json.load(f)

		# load feature model here
		med.feature_model = FeatureModel.from_file(feature_model_path)

		# load contexts here

		return med

	# TODO: incomplete
	# here run EMA
	def evaluate_context(self, resolution_model, num_experiments, max_run_length, num_repititions, num_processes):
		# make sure the user has first specified the model
		try:
			assert self._model
		except AttributeError:
			raise AttributeError('first specify the Mediator.model')

		resolution_model = (self.uncertainties, self.outcomes)
		model_info = copy.deepcopy(self.model)
		model_info['med_name'] = self.name

		data = model_executor.execute(model_info, resolution_model, num_experiments, max_run_length, num_repititions, num_processes)

		print('testing context')

	# TODO: incomplete
	def learn(self):
		pass
		# RUN XCS HERE

	# TODO: incomplete
	def predict(self, **kwargs):
		pass
		# CHECK KWARGS AGAINST LEARNED INFORMATION HERE

	# TODO: incomplete
	def explain(self):
		pass
		# MAKE HEATMAPS HERE
