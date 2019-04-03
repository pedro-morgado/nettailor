import os

# Download data
cmds = [
	'wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-devkit.tar.gz',
	'tar xfvz decathlon-1.0-devkit.tar.gz',
	'rm decathlon-1.0-devkit.tar.gz',
	'cd decathlon-1.0/data',
	'wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-data.tar.gz',
	'tar xf decathlon-1.0-data.tar.gz',
	'echo Extracting aircraft',
	'tar xf aircraft.tar',
	'echo Extracting cifar100',
	'tar xf cifar100.tar',
	'echo Extracting daimlerpedcls',
	'tar xf daimlerpedcls.tar',
	'echo Extracting dtd',
	'tar xf dtd.tar',
	'echo Extracting gtsrb',
	'tar xf gtsrb.tar',
	'echo Extracting omniglot',
	'tar xf omniglot.tar',
	'echo Extracting svhn',
	'tar xf svhn.tar',
	'echo Extracting ucf101',
	'tar xf ucf101.tar',
	'echo Extracting vgg-flowers',
	'tar xf vgg-flowers.tar',
	'rm *.tar.gz *.tar',
	'cd ..',
	'wget http://www.svcl.ucsd.edu/~morgado/nettailor/data/decathlon_mean_std.pickle',
]
os.system(' && '.join(cmds))


