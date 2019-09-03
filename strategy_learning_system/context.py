# Brodderick Rodriguez
# Auburn University - CSSE
# 25 Aug. 2019

import os
import copy
import json
from . import util


class Context:
	def __init__(self, name=None):
		dts = util.datetime_str()

		if name is not None:
			self.name = name
		else:
			self.name = dts

		self.created_on = dts
		self.resolution_model = []
		self.num_experiments = 0
		self.num_repititions = 0
		self.max_run_length = 0
		self.num_processes = 1
		self.synthesized_data = None
		self.learned_data = None

	# TODO: load synthesized data
	# TODO: load learned_data
	@staticmethod
	def load(context_path):
		meta_data_path = context_path + '/meta_data.json'

		ctx = Context()

		with open(meta_data_path, 'r') as f:
			ctx.__dict__ = json.load(f)

		return ctx

	def save(self, path):
		self.path = util.clean_dir('{}/{}'.format(path, self.name))
		meta_data_path = self.path + '/meta_data.json'

		if not os.path.exists(self.path):
			os.mkdir(self.path)

		with open(meta_data_path, 'w') as f:
			context_dict = copy.deepcopy(self.__dict__)
			f.write(json.dumps(context_dict))

	@property
	def synthesized_data(self):
		if not os.path.exits('{}/synthesized_data.zip'.format(self.path)):
			raise ValueError('this context has not been evaluated yet.')

		# TODO: open and parse data here

		return self._synthesized_data

	@synthesized_data.setter
	def synthesized_data(self, v):
		pass
		# check here if file exists

	@property
	def learned_data(self):
		return self._learned_data

	@learned_data.setter
	def learned_data(self, v):
		pass
		# check here if file exists
	
	
