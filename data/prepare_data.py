import os
import glob
import scipy.io as sio
import torchvision

# Download data
cmds = [
	'mkdir -p flowers', 
	'cd flowers',
	'wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
	'wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
	'wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat',
	'echo Extracting 102flowers.tgz',
	'tar xfz 102flowers.tgz',
	'cd ..',
	'wget http://www.svcl.ucsd.edu/~morgado/nettailor/data/voc12.tar.gz',
	'echo Extracting voc12.tar.gz',
	'tar xfz voc12.tar.gz',
	'rm voc12.tar.gz',
]
os.system(' && '.join(cmds))

# Prepare Flowers data with ImageFolder structure
images = sorted(glob.glob('flowers/jpg/*'))
labels = (sio.loadmat('flowers/imagelabels.mat')['labels'][0]-1).tolist()
train_idx = (sio.loadmat('flowers/setid.mat')['trnid'][0]-1).tolist()
val_idx = (sio.loadmat('flowers/setid.mat')['valid'][0]-1).tolist()
test_idx = (sio.loadmat('flowers/setid.mat')['tstid'][0]-1).tolist()

for lbl in list(set(labels)):
	os.system('mkdir -p flowers/train/{lbl}'.format(lbl=lbl))
	os.system('mkdir -p flowers/test/{lbl}'.format(lbl=lbl))

print('Preparing training images')
for idx in train_idx:
	src_fn, lbl = images[idx], labels[idx]
	os.system('mv {src_fn} flowers/train/{lbl}/{fn}'.format(src_fn=src_fn, lbl=lbl, fn=src_fn.split('/')[-1]))

for idx in val_idx:
	src_fn, lbl = images[idx], labels[idx]
	os.system('mv {src_fn} flowers/train/{lbl}/{fn}'.format(src_fn=src_fn, lbl=lbl, fn=src_fn.split('/')[-1]))

print('Preparing test images')
for idx in test_idx:
	src_fn, lbl = images[idx], labels[idx]
	os.system('mv {src_fn} flowers/test/{lbl}/{fn}'.format(src_fn=src_fn, lbl=lbl, fn=src_fn.split('/')[-1]))

os.system('rm flowers/102flowers.tgz flowers/imagelabels.mat flowers/setid.mat && rmdir flowers/jpg')

torchvision.datasets.svhn.SVHN('svhn', split='train', download=True)
torchvision.datasets.svhn.SVHN('svhn', split='test', download=True)