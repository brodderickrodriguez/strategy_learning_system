# Brodderick Rodriguez
# Auburn University - CSSE
# 08 Jan. 2020

# class Rule:
# 	def __init__(self, classifier=None):
# 		self.model_uncertainties = []
# 		self.environmental_uncertainties = []
# 		self.outcome = 0
# 		self.classifier = classifier
#
# 	def __str__(self):
# 		s = '\nIF:\t  {}\nWHEN: {}\nTHEN: {}\n'
# 		f_i = '\n\t{a} <= {fname} <= {b}'
# 		s_m, s_e = '', ''
#
# 		for f, r in self.model_uncertainties:
# 			s_m += f_i.format(a=r[0], b=r[1], fname=f.name)
#
# 		for f, r in self.environmental_uncertainties:
# 			s_e += f_i.format(a=r[0], b=r[1], fname=f.name)
#
# 		return s.format(s_m, s_e, self.outcome)
#
#
# 	def __repr__(self):
# 		return str(self)
#
# 	@staticmethod
# 	def from_xcsr_classifier(classifier, env_uncertainty, mod_uncertainty, bins):
# 		rule = Rule(classifier)
# 		rule.outcome = classifier.predicted_payoff
#
# 		def _get_true_value(f_range, uncertainty):
# 			r = lambda p: p * (uncertainty.upper_bound - uncertainty.lower_bound) + uncertainty.lower_bound
# 			a, b = r(f_range[0]), r(f_range[1])
#
# 			if isinstance(eu, IntegerParameter):
# 				a, b = int(a), int(b)
#
# 			return a, b
#
# 		def _get_bin_range(a):
# 			l = a - 1 if a > 0 else 0
# 			r = a if a != 0 else 1
# 			return bins[l], bins[r]
#
# 		for pred, eu in zip(classifier.predicate, env_uncertainty):
# 			p_true = _get_true_value(pred, eu)
# 			rule.environmental_uncertainties.append((eu, p_true))
#
# 		for act, mu in zip(classifier.action, mod_uncertainty):
# 			act_range = _get_bin_range(act)
# 			act_true = _get_true_value(act_range, mu)
# 			rule.model_uncertainties.append((mu, act_true))
#
# 		return rule




class Rule:
	pass

class RuleSet:
	def __init__(self):
		pass

	@staticmethod
	def from_experiment_data(df):
		pass
		print('here')
		print(df)

		rs = RuleSet()


		return rs

	@staticmethod
	def from_xcsr_data(classifiers):
		pass
