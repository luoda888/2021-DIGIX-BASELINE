import os
import sys
import time
import math
import torch
import argparse

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

from detector.config import cfg
from contextlib import redirect_stdout
from detector.dataset import get_train_dataset
from detector.models import OCR_DETECTOR, CRITERION, LR_Scheduler_Head
from detector.utils.files import save_checkpoint
from detector.utils import cal_text_score, cal_kernel_score, runningScore, AverageMeter


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # dataset
        self.trainloader = get_train_dataset(cfg)
        # model
        model = OCR_DETECTOR(cfg)
        model.set_mode('TRAIN')
        params_list = model.parameters()
        optimizer = torch.optim.SGD(params_list, lr=cfg.MODEL.TRAIN.OPTIMIZER.LR,
                                    momentum=cfg.MODEL.TRAIN.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.MODEL.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        self.criterion = CRITERION[cfg.MODEL.LOSS.ARCH]()
        self.model, self.optimizer = model, optimizer

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).cuda()
            torch.backends.cudnn.benchmark = True

        print("=> Training Mode: FROM-SCRATCH")
        self.start_epoch = 0

        # lr schedule
        self.num_epochs = math.ceil(cfg.MODEL.TRAIN.SUM_STEP / len(self.trainloader))
        print(self.num_epochs)
        print(self.num_epochs // 3)
        self.schedule = LR_Scheduler_Head(mode=cfg.MODEL.TRAIN.OPTIMIZER.MODE,
                                          base_lr=cfg.MODEL.TRAIN.OPTIMIZER.LR,
                                          num_epochs=self.num_epochs,
                                          iters_per_epoch=len(self.trainloader),
                                          lr_step=1)
        self.best_pred = 0.0

    def training(self, epoch, model_save_path):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
        self.model.train()

        end = time.time()
        for i, (image, target, training_mask) in enumerate(self.trainloader):
            image = image.cuda()
            target = target.cuda()
            training_mask = training_mask.cuda()
            data_time.update(time.time() - end)

            self.schedule(self.optimizer, i, epoch, self.best_pred)
            outputs = self.model(image)
            loss = self.criterion(outputs, target, training_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), image.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            if i % 100 == 0:
                score_text = cal_text_score(outputs[:, 0, :, :], target[:, 0, :, :], training_mask, running_metric_text)
                score_kernel = cal_kernel_score(outputs[:, 1:, :, :], target[:, 1:, :, :],
                                                target[:, 0, :, :], training_mask, running_metric_kernel)
                info = 'batch: {:4d}/{:4d}. loss: {:.4f}. iou-t: {:.3f}. iou-k: {:.3f}. accuracy: {:.3f}'.format(
                    i + 1, len(self.trainloader), losses.avg, score_text['Mean IoU'],
                    score_kernel['Mean IoU'], score_text['Mean Acc']
                )

                print(info)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, model_save_path, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Noah OCR Detection.')
    parser.add_argument('--config', type=str, default=r'detector/config/resnet50.yaml', help='config yaml file.')
    parser.add_argument('--cuda', type=str, default='0', help='cuda devices id.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    cfg.merge_from_file(args.config)
    cfg.freeze()

    if not os.path.isdir(cfg.MODEL.TRAIN.SAVE_PATH):
        os.makedirs(cfg.MODEL.TRAIN.SAVE_PATH)

    with open(os.path.join(cfg.MODEL.TRAIN.SAVE_PATH, 'config.txt'), 'w') as f:
        with redirect_stdout(f):
            print(cfg.dump())

    print('*****model info*****:')
    print('=> 1.backbone: ', cfg.MODEL.BACKBONE.ARCH)
    print('=> 2.neck: ', cfg.MODEL.NECK.ARCH)
    print('=> 3.head: ', cfg.MODEL.HEAD.ARCH)

    trainer = Trainer(cfg)
    print('*****start training*****:')
    print('=> 1.Starting Epoch: ', trainer.start_epoch)
    print('=> 2.Total Epoch: ', trainer.num_epochs)
    for epoch in range(trainer.start_epoch, trainer.num_epochs):
        trainer.training(epoch, cfg.MODEL.TRAIN.SAVE_PATH)

'''bash
python detector/train.py
'''
