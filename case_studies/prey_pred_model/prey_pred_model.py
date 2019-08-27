# Brodderick Rodriguez
# Auburn University - CSSE
# 27 Aug. 2019

import strategy_learning_system as sls

def define_problem():
	inter = sls.Interpreter(name='prey_pred')

	model_dir = '/Users/bcr/Dropbox/Projects/CODE/netlogo/prey_predator_nlogo'
	model_name = 'preypred.nlogo'
	inter.model = (model_dir, model_name)


	return inter


interpreter = define_problem()
print(interpreter)