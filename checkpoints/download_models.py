import os

# Download models
cmds = [
	'wget http://www.svcl.ucsd.edu/~morgado/nettailor/data/wide_resnet26.pth.tar',
	'wget http://www.svcl.ucsd.edu/~morgado/nettailor/data/decathlon_models.tar.gz',
	'tar xfvz decathlon_models.tar.gz',
]
os.system(' && '.join(cmds))