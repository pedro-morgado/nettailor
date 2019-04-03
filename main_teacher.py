import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import resnet
import proj_utils
import dataloaders
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Taskonomy Training')

parser.add_argument('--task', default='cifar100', help='task to train')
parser.add_argument('--arch', default='resnet34', help='teacher architecture (default: resnet34)')
parser.add_argument('--model-dir', default='experiments/default', help='model directory')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--lr-decay-epochs', default=30, type=int, help='number of epochs for each lr decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', default=10, type=int, help='print frequency (default: 10 iter)')
parser.add_argument('--eval-freq', default=5, type=int, help='print frequency (default: 5 epochs)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--log2file', action='store_true', help='log output to file (under model_dir/train.log)')

args = parser.parse_args()
proj_utils.prep_output_folder(args.model_dir, args.evaluate)
DEVICE = torch.device("cuda:0")

torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = True
np.random.seed(args.seed)

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
            shuffle=False, 
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
    model = eval('resnet.{}(pretrained=True, num_classes={})'.format(args.arch, num_classes))
    model = model.to(DEVICE)

    logger.add_line("="*30+"   Model   "+"="*30)
    logger.add_line(str(model))
    logger.add_line("="*30+"   Parameters   "+"="*30)
    logger.add_line(proj_utils.parameter_description(model))

    #Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if mode == 'train':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
            momentum=args.momentum, weight_decay=args.weight_decay)
    
    ############################ TRAIN #########################################
    if mode == 'train':
        for epoch in range(args.epochs):
            # Train for one epoch
            logger.add_line("="*30+"   Train (Epoch {})   ".format(epoch)+"="*30)
            optimizer = proj_utils.adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay_epochs, logger)
            train(train_loader, model, criterion, optimizer, epoch, logger)

            if epoch % args.eval_freq == args.eval_freq-1:
                # Evaluate on validation set
                logger.add_line("="*30+"   Valid (Epoch {})   ".format(epoch)+"="*30)
                err, acc, run_time = validate(val_loader, model, criterion, logger, epoch)
                
                # remember best err and save checkpoint
                proj_utils.save_checkpoint(
                    args.model_dir, 
                    {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'err': err,
                     'acc': acc})

    ############################ EVAL #########################################
    elif mode == 'eval':
        fn = args.model_dir + '/checkpoint.pth.tar'
        model.load_state_dict(torch.load(fn)['state_dict'])
        err, acc, run_time = validate(test_loader, model, criterion, logger)

    logger.add_line('='*30+'  COMPLETED  '+'='*30)
    logger.add_line('[RUN TIME] {time.avg:.3f} sec/sample'.format(time=run_time))
    logger.add_line('[FINAL] {name:<30} {loss:.7f}'.format(name='crossentropy', loss=err))
    logger.add_line('[FINAL] {name:<30} {acc:.7f}'.format(name='accuracy', acc=acc))


def train(data_loader, model, criterion, optimizer, epoch, logger):
    batch_time = proj_utils.AverageMeter()
    data_time = proj_utils.AverageMeter()
    loss_avg = proj_utils.AverageMeter()
    acc_avg = proj_utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if images.size(0) != args.batch_size:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        logit, _ = model(images)
        loss = criterion(logit, labels)
        loss_avg.update(loss.item(), images.size(0))
        acc = proj_utils.accuracy(logit, labels)
        acc_avg.update(acc.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i+1 == len(data_loader):
            logger.add_line(
                "TRAIN [{:5}][{:5}/{:5}] | Time {:16} Data {:16} Accuracy {:18} Loss {:16}".format(
                    str(epoch), str(i), str(len(data_loader)), 
                    "{t.val:.3f} ({t.avg:.3f})".format(t=batch_time),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=data_time),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=acc_avg),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=loss_avg),
                    ))

def validate(data_loader, model, criterion, logger, epoch=None):
    batch_time = proj_utils.AverageMeter()
    loss_avg = proj_utils.AverageMeter()
    acc_avg = proj_utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    image_ids, preds = [], []
    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # compute output
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss_avg.update(loss.item(), images.size(0))
            acc = proj_utils.accuracy(logits, labels)
            acc_avg.update(acc.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end, images.size(0))
            end = time.time()

            if i % args.print_freq == 0 or i+1 == len(data_loader):
                logger.add_line(
                    "Test [{:5}][{:5}/{:5}] | Time {:20} Accuracy {:20} Loss {:20}".format(
                        str(epoch), str(i), str(len(data_loader)), 
                        "{t.val:.3f} ({t.avg:.3f})".format(t=batch_time),
                        "{t.val:.3f} ({t.avg:.3f})".format(t=acc_avg),
                        "{t.val:.3f} ({t.avg:.3f})".format(t=loss_avg),
                        ))
    return loss_avg.avg, acc_avg.avg, batch_time

if __name__ == '__main__':
    main()
