# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

from . import xcsr_interface
from . import mlp_hac_interface


_algorithm_interface_map = {'xcsr': xcsr_interface, 'mlp_hac': mlp_hac_interface}


def learn(mediator, context, algorithm):
	if algorithm not in _algorithm_interface_map:
		raise ValueError('algorithm not known: {}'.format(algorithm))
	else:
		results = _algorithm_interface_map[algorithm].learn(mediator, context)
		return results
