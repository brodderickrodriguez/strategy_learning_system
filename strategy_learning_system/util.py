# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import datetime


def clean_dir_path(dir):
	d = dir.replace('//', '/')
	return d


def datetime_str():
	dts = str(datetime.datetime.now())
	for e in ['.', ':', '-', ' ']:
		dts = dts.replace(e, '_')
	return dts
