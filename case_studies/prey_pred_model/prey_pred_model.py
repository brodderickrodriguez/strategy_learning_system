# Brodderick Rodriguez
# Auburn University - CSSE
# 27 Aug. 2019

import strategy_learning_system as sls


def define_problem():
	inter = sls.Interpreter(name='prey_pred')

	root = '/Users/bcr/Dropbox/Projects/CODE'

	model_dir = root + '/netlogo/prey_predator_nlogo'
	model_name = 'preypred.nlogo'
	inter.model = (model_dir, model_name)

	inter.save_location = root + '/Python/strategy_learning_system/case_studies/prey_pred_model'
	return inter



interpreter = define_problem()
interpreter.save()
print(interpreter)