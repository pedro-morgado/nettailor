import os, sys

# Provided dataloaders: 'flowers', 'svhn', 'voc12'.
TASK = 'flowers' 

# Supported backbone CNNs: 'resnet18', 'resnet34', 'resnet50'
BACKBONE = 'resnet34'

# Select GPU id
GPU = 0

########################## TRAIN TEACHER ############################################### 

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.01
LR_EPOCHS = 20
WEIGHT_DECAY = 0.0005

teacher_dir = 'checkpoints/{task}/{arch}'.format(task=TASK, arch=BACKBONE)
cmd = ("CUDA_VISIBLE_DEVICES={gpu} "
	   "python train_teacher.py "
	   "--task {task} "
	   "--model-dir {model_dir} "
	   "--arch {arch} "
	   "--epochs {epochs} "
	   "--batch-size {bs} "
	   "--lr {lr} "
	   "--lr-decay-epochs {lr_epochs} "
	   "--weight-decay {wd} "
	   "--workers 4 "
	   "--log2file ").format(
			gpu=GPU, task=TASK, arch=BACKBONE, model_dir=teacher_dir,
			epochs=EPOCHS, bs=BATCH_SIZE, lr=LR, lr_epochs=LR_EPOCHS, wd=WEIGHT_DECAY)

print(cmd)
os.system(cmd)

print(cmd + ' --evaluate')
os.system(cmd + ' --evaluate')

########################## TRAIN STUDENT ############################################### 

COMPLEXITY_COEFF = 0.3
TEACHER_COEFF = 10.0
MAX_SKIP = 3

BATCH_SIZE = 32
EPOCHS = 50
LR_EPOCHS = 20
LR = 0.1

teacher_fn = teacher_dir + '/checkpoint.pth.tar'
full_model_fn = 'checkpoints/{task}/nettailor-{backbone}-{max_skip}Skip-D{teacher}-G{complexity}'.format(
	task=TASK, backbone=BACKBONE, max_skip=MAX_SKIP, teacher=TEACHER_COEFF, complexity=COMPLEXITY_COEFF)
cmd = ("CUDA_VISIBLE_DEVICES={gpu} "
	   "python train_student.py "
	   "--task {task} "
	   "--model-dir {full_model_fn} "
	   "--teacher-fn {teacher_fn} "
	   "--backbone {backbone} "
	   "--max-skip {max_skip} "
	   "--complexity-coeff {complexity} "
	   "--teacher-coeff {teacher} "
	   "--epochs {epochs} "
	   "--batch-size {bs} "
	   "--lr {lr} "
	   "--lr-decay-epochs {lr_epochs} "
	   "--weight-decay {wd} "
	   "--workers 4 "
	   "--log2file ").format(
			gpu=GPU, task=TASK, full_model_fn=full_model_fn, backbone=BACKBONE, teacher_fn=teacher_fn, max_skip=MAX_SKIP,
			complexity=COMPLEXITY_COEFF, teacher=TEACHER_COEFF, epochs=EPOCHS, bs=BATCH_SIZE, lr=LR, lr_epochs=LR_EPOCHS, wd=WEIGHT_DECAY)

print(cmd)
os.system(cmd)

print(cmd + " --evaluate")
os.system(cmd + " --evaluate")

########################## PRUNE STUDENT AND RETRAIN ############################################### 

NUM_BLOCKS_PRUNED = 7
PROXY_PRUNING_THRESHOLD = 0.05
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.01
LR_EPOCHS = 20

pruned_model_fn = 'checkpoints/{task}/nettailor-{backbone}-{max_skip}Skip-D{teacher}-G{complexity}-Pruned{thr}'.format(
	task=TASK, backbone=BACKBONE, max_skip=MAX_SKIP, teacher=TEACHER_COEFF, complexity=COMPLEXITY_COEFF, thr=NUM_BLOCKS_PRUNED)
cmd = ("CUDA_VISIBLE_DEVICES={gpu} "
	   "python train_student.py "
	   "--task {task} "
	   "--model-dir {pruned_model_fn} "
	   "--full-model-dir {full_model_fn} "
	   "--n-pruning-universal {thr} "
	   "--thr-pruning-proxy {adapt_thr} "
	   "--teacher-fn {teacher_fn} "
	   "--backbone {backbone} "
	   "--max-skip {max_skip} "
	   "--complexity-coeff 0.0 "
	   "--teacher-coeff {teacher} "
	   "--epochs {epochs} "
	   "--batch-size {bs} "
	   "--lr {lr} "
	   "--lr-decay-epochs {lr_epochs} "
	   "--workers 4 "
	   "--log2file ").format(
			gpu=GPU, task=TASK, pruned_model_fn=pruned_model_fn, full_model_fn=full_model_fn, backbone=BACKBONE, teacher_fn=teacher_fn, 
			thr=NUM_BLOCKS_PRUNED, adapt_thr=PROXY_PRUNING_THRESHOLD, max_skip=MAX_SKIP, teacher=TEACHER_COEFF, epochs=EPOCHS, bs=BATCH_SIZE, lr=LR, lr_epochs=LR_EPOCHS)

print(cmd)
os.system(cmd)

print(cmd + " --evaluate")
os.system(cmd + " --evaluate")
