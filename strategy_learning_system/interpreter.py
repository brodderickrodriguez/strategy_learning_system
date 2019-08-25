# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import os
import datetime


class Interpreter:
	def __init__(self, name=None, from_file=False):
		self.name = name

		if not from_file:
			self.created_on = datetime.datetime.now()

	def __str__(self):
		return self.name

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

		# make sure that the model_dir is valid
		assert os.path.exists(model_dir)

		# make sure that the model_file exists
		assert os.path.isfile(model_dir + model_name)

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
			raise AttributeError('first specify the Experiment.save_location')
		
		pass
		# SAVE STUFF HERE

	# TODO: incomplete
	@staticmethod
	def load(file):
		return Experiment(from_file=True)

	# TODO: incomplete
	def execute_model(self, number_experiments, max_run_length, replications, processes):
		# make sure the user has first specified the model
		try:
			assert self._model
		except:
			raise AttributeError('first specify the Experiment.model')

		pass
		# RUN EMA HERE

	# TODO: incomplete
	def learn(self):
		pass
		# RUN XCS HERE

	# TODO: incomplete
	def predict(self, **kwargs):
		pass
		# CHECK KWARGS AGAINST LEARNED INFORMATION HERE

	# TODO: incomplete
	def interpret(self):
		pass
		# MAKE HEATMAPS HERE


if __name__ == '__main__':
	e = Experiment(' test name 1 ')
	
	e.name = ' test name 2 '

	e.model = './', 'util.py'

	print(e.model)

















