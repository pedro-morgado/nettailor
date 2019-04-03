import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import resnet
import nettailor
import dataloaders
import proj_utils
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch Taskonomy Training')

parser.add_argument('--model-dir', metavar='MODEL_DIR',
                    help='model directory')
parser.add_argument('--task', metavar='TASK',
                    help='task to train')
parser.add_argument('--teacher-fn', metavar='TEACHER_CHECKPOINT',
                    help='Teacher checkpoint filename')
parser.add_argument('--backbone', metavar='BACKBONE', default='resnet34',
                    help='backbone model architecture: ' + ' (default: resnet34)')
parser.add_argument('--max-skip', default=3, type=int)
parser.add_argument('--complexity-coeff', default=1.0, type=float)
parser.add_argument('--teacher-coeff', default=10.0, type=float)

parser.add_argument('--epochs', default=50, type=int, 
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='BS', help='batch size')
parser.add_argument('--lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay-epochs', default=20, type=int, 
                    metavar='LR_EPOCHS', help='number of epochs for each lr decay')
parser.add_argument('--momentum', default=0.9, type=float, 
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume-path', default='')
parser.add_argument('--full-model-dir', metavar='MODEL_DIR', default='',)
parser.add_argument('--n-pruning-universal', metavar='THR', default=0, type=float)
parser.add_argument('--thr-pruning-proxy', metavar='THR', default=0.075, type=float)


parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=5, type=int,
                    metavar='N', help='eval frequency (default: 5)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=0, type=int, 
                    help='random seed')
parser.add_argument('--log2file', action='store_true',
                    help='log output to file (under model_dir/train.log)')
parser.add_argument('--log2tb', action='store_true',
                    help='log output to tensorboard')

args = parser.parse_args()
DEVICE = torch.device("cuda:0")
proj_utils.prep_output_folder(args.model_dir, args.evaluate)

def main():
    mode = 'train' if not args.evaluate else 'eval'
    logger = proj_utils.Logger(args.log2file, mode=mode, model_dir=args.model_dir)

    # Args
    logger.add_line(str(datetime.datetime.now()))
    logger.add_line("="*30+"   Arguments   "+"="*30)
    for k in args.__dict__:
        logger.add_line(' {:30}: {}'.format(k, str(args.__dict__[k])))

    # Data
    if mode == 'train':
        train_loader = dataloaders.get_dataloader(
            dataset=args.task, 
            batch_size=args.batch_size,
            shuffle=True, 
            mode=mode,
            num_workers=args.workers)

        val_loader = dataloaders.get_dataloader(
            dataset=args.task,
            batch_size=args.batch_size, 
            shuffle=True, 
            mode='eval', 
            num_workers=args.workers)
        num_classes = train_loader.dataset.num_classes

    elif mode == 'eval':
        test_loader = dataloaders.get_dataloader(
            dataset=args.task, 
            batch_size=args.batch_size, 
            shuffle=False, 
            mode=mode, 
            num_workers=args.workers)
        num_classes = test_loader.dataset.num_classes

    # Model
    model = nettailor.create_model(
        num_classes=num_classes, 
        max_skip=args.max_skip,
        backbone=args.backbone
    )
    backbone_tensors = get_backbone_tensors(model)

    logger.add_line("="*30+"   Model   "+"="*30)
    logger.add_line(str(model))
    logger.add_line("="*30+"   Parameters   "+"="*30)
    logger.add_line(proj_utils.parameter_description(model))

    # Teacher
    teacher = eval('resnet.{}(num_classes={})'.format(args.backbone.split('_')[0], num_classes))
    teacher.freeze()
    logger.add_line("\n"+"="*30+"   Teacher   "+"="*30)
    logger.add_line("Loading pretrained teacher from: " + args.teacher_fn)
    teacher.load_pretrained(args.teacher_fn)

    #Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if mode == 'train':
        parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'proxies' in n], 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'alphas_params' in n], 'lr': args.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ]
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum)
        del parameters

    # Resume from a checkpoint
    start_epoch = 0
    if mode == 'eval':
        logger.add_line("\n"+"="*30+"   Checkpoint   "+"="*30)
        logger.add_line("Loading checkpoint from: " + args.model_dir)
        proj_utils.load_checkpoint(model, model_dir=args.model_dir)
    if mode == 'train' and len(args.full_model_dir) > 0:
        proj_utils.load_checkpoint(model, model_dir=args.full_model_dir)
    if mode == 'train' and len(args.resume_path) > 0:
        _, start_epoch = proj_utils.load_checkpoint(model, model_fn=args.resume_path, optimizer=optimizer, outps=['epoch'])

    model.to(DEVICE)
    teacher = teacher.to(DEVICE)

    ############################ TRAIN #########################################
    if mode == 'train':
        if len(args.full_model_dir) > 0: # Block pruning
            model.threshold_alphas(num_global=int(args.n_pruning_universal), thr_proxies=args.thr_pruning_proxy)
        logger.add_line("\n" + "="*30+"   Model Stats   "+"="*30)
        logger.add_line(model.stats())

        for ii, epoch in enumerate(range(start_epoch, args.epochs)):
            # Train for one epoch
            logger.add_line("="*30+"   Train (Epoch {})   ".format(epoch)+"="*30)
            optimizer = proj_utils.adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay_epochs, logger)
            train(train_loader, model, teacher, criterion, optimizer, epoch, logger)

            if epoch % args.eval_freq == args.eval_freq-1 or epoch == args.epochs-1:
                # Evaluate on validation set
                logger.add_line("="*30+"   Valid (Epoch {})   ".format(epoch)+"="*30)
                err, acc, run_time = validate(val_loader, model, teacher, criterion, logger, epoch)
                
                # Save checkpoint
                proj_utils.save_checkpoint(args.model_dir, {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'keep_flags': model.get_keep_flags(),
                        'acc': acc,
                        'xent': err
                    }, ignore_tensors=backbone_tensors)

                logger.add_line(model.alphas_and_distances())
                if epoch != args.epochs-1:
                    del acc, err

    ############################ EVAL #########################################
    elif mode == 'eval':
        logger.add_line("="*30+"   Evaluation   "+"="*30)
        err, acc, run_time = validate(test_loader, model, teacher, criterion, logger)

    logger.add_line('='*30+'  COMPLETED  '+'='*30)
    logger.add_line(model.stats())
    logger.add_line('[RUN TIME] {time.avg:.3f} sec/sample'.format(time=run_time))
    logger.add_line('[FINAL] {name:<30} {loss:.7f}'.format(name=args.task+'/crossentropy', loss=err))
    logger.add_line('[FINAL] {name:<30} {acc:.7f}'.format(name=args.task+'/accuracy', acc=acc))


def train(data_loader, model, teacher, criterion, optimizer, epoch, logger):
    batch_time = proj_utils.AverageMeter()
    data_time = proj_utils.AverageMeter()
    loss_avg = proj_utils.AverageMeter()
    complexity_avg = proj_utils.AverageMeter()
    teacher_avg = proj_utils.AverageMeter()
    acc_avg = proj_utils.AverageMeter()

    # switch to train mode
    model.train()
    teacher.eval()

    logger.add_line('Complexity coefficient:   {}'.format(args.complexity_coeff))
    logger.add_line('Teacher coefficient:      {}'.format(args.teacher_coeff))

    l2dist = nn.MSELoss()
    end = time.time()
    for i, (images, labels) in enumerate(data_loader):
        if images.size(0) != args.batch_size:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute outputs
        logit, ends_model = model(images)

        # Classification loss
        loss = criterion(logit, labels)
        loss_avg.update(loss.item(), images.size(0))
        acc = proj_utils.accuracy(logit, labels)
        acc_avg.update(acc.item(), images.size(0))

        # Teacher supervision
        with torch.no_grad():
            teacher_logit, ends_teacher = teacher(images)
        teacher_loss = l2dist(logit, teacher_logit.detach())
        for e1, e2 in zip(ends_model, ends_teacher):
            if e1 is not None:
                teacher_loss += l2dist(e1, e2.detach())
        teacher_loss /= float(len(ends_model))
        teacher_avg.update(teacher_loss.item(), images.size(0))

        # Model complexity
        complexity = model.expected_complexity()
        complexity_avg.update(complexity.item(), 1)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss = loss + args.complexity_coeff * complexity + args.teacher_coeff * teacher_loss
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.add_line(
                "TRAIN [{:5}][{:5}/{:5}] | Time {:6} Data {:6} Acc {:22} Loss {:16} Complexity {:7} Teacher Sup {:7}".format(
                    str(epoch), str(i), str(len(data_loader)), 
                    "{t.avg:.3f}".format(t=batch_time),
                    "{t.avg:.3f}".format(t=data_time),
                    "{t.val:.3f} (Avg: {t.avg:.3f})".format(t=acc_avg),
                    "{t.val:.3f} (Avg: {t.avg:.3f})".format(t=loss_avg),
                    "{t.val:.3f}".format(t=complexity_avg),
                    "{t.val:.3f}".format(t=teacher_avg)
                ))


def validate(data_loader, model, teacher, criterion, logger, epoch=None):
    batch_time = proj_utils.AverageMeter()
    loss_avg = proj_utils.AverageMeter()
    acc_avg = proj_utils.AverageMeter()
    loss_teacher_avg = proj_utils.AverageMeter()
    acc_teacher_avg = proj_utils.AverageMeter()
    complexity_avg = proj_utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    teacher.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward data through student
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss_avg.update(loss.item(), images.size(0))
            acc = proj_utils.accuracy(logits, labels)
            acc_avg.update(acc.item(), images.size(0))
            complexity = model.expected_complexity()
            complexity_avg.update(complexity.item(), 1)
            
            # Forward data through teacher
            logits_teacher, _ = teacher(images)
            loss_teacher = criterion(logits_teacher, labels)
            loss_teacher_avg.update(loss_teacher.item(), images.size(0))
            acc_teacher = proj_utils.accuracy(logits_teacher, labels)
            acc_teacher_avg.update(acc_teacher.item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end, images.size(0))
            end = time.time()
            
            if i % args.print_freq == 0 or i == len(data_loader)-1:
                logger.add_line(
                    "Test [{:5}][{:5}/{:5}] | Time {:5} | Acc {:8} XEnt {:6} Complexity {:6} | Teacher: Acc {:8} XEnt {:6} ".format(
                        str(epoch), str(i), str(len(data_loader)), 
                        "{t.avg:.3f}".format(t=batch_time),
                        "{t.avg:.3f}".format(t=acc_avg),
                        "{t.avg:.3f}".format(t=loss_avg),
                        "{t.avg:.3f}".format(t=complexity_avg),
                        "{t.avg:.3f}".format(t=acc_teacher_avg),
                        "{t.avg:.3f}".format(t=loss_teacher_avg),
                    ))

    return loss_avg.avg, acc_avg.avg, batch_time


def get_backbone_tensors(model):
    tensors = {}
    for k in model.state_dict():
        if not ('proxies' in k or 'classifier' in k or 'alphas' in k or 'running' in k or 'tracked' in k or 'ends_bn' in k):
            if k.startswith('layer'):
                k_ckp = '.'.join(k.split('.')[:2] + k.split('.')[3:])
            elif k.startswith('classifier'):
                k_ckp = 'linear.{}'.format(k.split('.')[-1])
            else:
                k_ckp = k
            tensors[k_ckp] = k
    return tensors


if __name__ == '__main__':
    main()
