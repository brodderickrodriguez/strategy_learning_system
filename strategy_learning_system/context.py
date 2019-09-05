# Brodderick Rodriguez
# Auburn University - CSSE
# 25 Aug. 2019

import os
import copy
from . import util


class Context:
	def __init__(self, name=None):
		dts = util.datetime_str()

		self.name = name if name is not None else dts
		self.created_on = dts
		self.resolution_model = []
		self.num_experiments = 0
		self.num_replications = 0
		self.max_run_length = 0
		self.num_processes = 1
		self.synthesized_data = None
		self.learned_data = None
		self.data_path = None

	def __str__(self):
		return 'Context: {}'.format(self.name)

	def __repr__(self):
		return self.__str__()

	def __eq__(self, o):
		return self.name == o.name

	def add_feature(self, f):
		if f not in self.resolution_model:
			self.resolution_model.append(f)

	@property
	def synthesized_data(self):
		if not os.path.exits('{}/synthesized_data.zip'.format(self.dir_path)):
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
