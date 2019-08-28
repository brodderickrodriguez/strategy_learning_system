# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import os
import datetime
import pickle
import ema_workbench
from . import model_executor
import copy


IntegerParameter = ema_workbench.IntegerParameter

TimeSeriesOutcome = ema_workbench.TimeSeriesOutcome


class Interpreter:
	def __init__(self, name=None, from_file=False):
		self.name = name

		if not from_file:
			self.created_on = datetime.datetime.now()

		# TODO: replace these using only the feature model
		self.uncertainties = None
		self.outcomes = None
		self.feature_model = None

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
		if model_dir[-1] != '/':
			model_dir += '/'

		model_dir = (model_dir + '/').replace('//', '/')

		try:
			# make sure that the model_dir is valid
			assert os.path.exists(model_dir)

			# make sure that the model_file exists
			assert os.path.isfile(model_dir + model_name)
		except AssertionError:
			raise AttributeError('Interpreter.model is not a valid file')

		# finally save the model info
		self._model = {'dir': model_dir, 'name': model_name}

	@property
	def save_location(self):
		return self._save_location

	@save_location.setter
	def save_location(self, save_dir):
		# make sure save_dir is valid
		assert os.path.exists(save_dir)

		# save the save_location
		self._save_location = save_dir

	# TODO: incomplete
	def save(self):
		# make sure the user has specified the save_location before saving
		try:
			assert self._save_location
		except AttributeError:
			raise AttributeError('first specify the Interpreter.save_location')
		
		pass
		# SAVE STUFF HERE

	# TODO: incomplete
	@staticmethod
	def load(file):
		return Experiment(from_file=True)

	# TODO: incomplete
	# here run EMA
	def evaluate_context(self, resolution_model, num_experiments, max_run_length, num_repititions, num_processes):
		# make sure the user has first specified the model
		try:
			assert self._model
		except:
			raise AttributeError('first specify the Interpreter.model')

		resolution_model = (self.uncertainties, self.outcomes)
		model_info = copy.deepcopy(self.model)
		model_info['inter_name'] = self.name

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
