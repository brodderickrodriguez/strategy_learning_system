# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import xcsr
from . import xcsr_interface
from . import ann_hac_interface


# TODO: delete before publishing
# due to pickling this has to be here
class GenericConfiguration(xcsr.Configuration):
	pass


def learn(mediator, cxt, algorithm):
	if algorithm == 'xcsr':
		return xcsr_interface.learn(mediator, cxt)
	elif algorithm == 'ann_hac':
		return ann_hac_interface.learn(mediator, cxt)
	else:
		raise ValueError('algorithm not known: {}'.format(algorithm))
