# Brodderick Rodriguez
# Auburn University - CSSE
# 25 Aug. 2019


class Rule:
	def __init__(self, classifier=None):
		self.model_uncertainties = []
		self.environmental_uncertainties = []
		self.outcome = 0
		self.confidence = 1
		self.experience = 1
		self.classifier = classifier

	def __str__(self):
		s = '\nIF:\t  {0}\nWHEN: {1}\nTHEN: {2:0.3f}\n(conf.: {3:0.3f}, exp.: {4})\n'
		f_i = '\n\t{a} <= {fname} <= {b}'
		s_m, s_e = '', ''

		for f, r, _ in self.model_uncertainties:
			s_m += f_i.format(a=r[0], b=r[1], fname=f.name)

		for f, r, _ in self.environmental_uncertainties:
			s_e += f_i.format(a=r[0], b=r[1], fname=f.name)

		return s.format(s_m, s_e, self.outcome, self.confidence, self.experience)

	def __repr__(self):
		return str(self)

	def __lt__(self, other):
		sv = self.outcome * self.confidence
		ov = other.outcome * other.confidence
		return sv < ov

	def __eq__(self, other):
		return self.model_uncertainties == other.model_uncertainties and \
			   self.environmental_uncertainties == other.environmental_uncertainties
