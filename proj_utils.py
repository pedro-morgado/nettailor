import sys, os
import numpy as np
import torch

def prep_output_folder(model_dir, evaluate):
    if evaluate:
        assert os.path.isdir(model_dir)
    else:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
    
def parameter_description(model):
    desc = ''
    for n, p in model.named_parameters():
        desc += "{:50} | {:10} | {:30} | {}\n".format(
            n, 'Trainable' if p.requires_grad else 'Frozen' , 
            str(p.size()), str(np.prod(p.size())))
    return desc

def accuracy(predictions, targets, axis=1):
    batch_size = predictions.size(0)
    predictions = predictions.max(axis)[1].type_as(targets)
    hits = predictions.eq(targets)
    acc = 100. * hits.sum().float() / float(batch_size)
    return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, log2file=False, mode='train', model_dir=None):
        if log2file:
            assert model_dir is not None
            fn = os.path.join(model_dir, '{}.log'.format(mode))
            self.fp = open(fn, 'w')
        else:
            self.fp = sys.stdout

    def add_line(self, content):
        self.fp.write(content+'\n')
        self.fp.flush()

def load_checkpoint(model, model_dir=None, model_fn=None, tensors=None, optimizer=None, outps=None):
    assert (model_dir is not None and model_fn is None) or (model_dir is None and model_fn is not None)
    fn = model_fn if model_fn is not None else os.path.join(model_dir, 'checkpoint.pth.tar')
    checkpoint = torch.load(fn)

    state = model.state_dict()

    n_loaded, n_ignored = 0, 0
    loaded_tensors = []
    for k_ckp in checkpoint['state_dict']:
        if tensors is not None and k_ckp not in tensors:
            n_ignored += 1
            continue
        k_st = tensors[k_ckp] if tensors is not None else k_ckp
        if k_st in state:
            state[k_st] = checkpoint['state_dict'][k_ckp].clone()
            n_loaded += 1
            loaded_tensors.append(k_st)
        else:
            n_ignored += 1
            
    print('Loading checkpoint: {}\n - Tensors loaded {}\n - Tensors ignored {}.\n'.format(fn, n_loaded, n_ignored))
    model.load_state_dict(state)

    if 'keep_flags' in checkpoint:
        model.load_keep_flags(checkpoint['keep_flags'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if outps is None:
        return loaded_tensors
    else:
        return loaded_tensors, tuple([checkpoint[k] for k in outps])

def save_checkpoint(model_dir, state, ignore_tensors=None):
    checkpoint_fn = os.path.join(model_dir, 'checkpoint.pth.tar')
    if ignore_tensors is not None:
        for p in ignore_tensors.values():
            if p in state['state_dict']:
                del state['state_dict'][p]
    torch.save(state, checkpoint_fn)


def adjust_learning_rate(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7, logger=None):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    logger.add_line('Learning rate:            {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

    