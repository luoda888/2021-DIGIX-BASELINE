import torch
import torch.nn as nn
import numpy as np

__all__ = ['DiceLoss']


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def _ohem_batch(self, texts, gt_texts, training_masks):
        '''
        :param scores: [N * H * W]
        :param gt_texts:  [N * H * W]
        :param training_masks: [N * H * W]
        :return: [N * H * W]
        '''
        selected_masks = []

        for text, gt_text, training_mask in zip(texts, gt_texts, training_masks):
            selected_mask = self._ohem_single(text, gt_text, training_mask)
            selected_masks.append(selected_mask)

        selected_masks = torch.cat(selected_masks, dim=0)
        return selected_masks

    def _ohem_single(self, text, gt_text, training_mask):

        pos_mask = (gt_text > 0.5) & (training_mask > 0.5)
        pos_num = torch.sum(pos_mask.type(torch.int32))

        neg_mask = (gt_text <= 0.5)
        neg_num = torch.sum(neg_mask.type(torch.int32))
        neg_num = torch.min(pos_num * 3, neg_num)

        if neg_num == 0:
            select_mask = torch.unsqueeze(training_mask, 0)
        else:
            neg_score = text[neg_mask].reshape(-1)
            neg_score_sorted, _ = torch.sort(neg_score, descending=True)
            neg_threshold = neg_score_sorted[neg_num - 1]
            select_mask = ((text >= neg_threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
            select_mask = torch.unsqueeze(select_mask.float(), dim=0)
        return select_mask

    def _dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)

        batch_size = input.shape[0]
        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1)
        mask = mask.contiguous().view(batch_size, -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1)
        c = torch.sum(target * target, 1)
        d = (2 * a) / (b + c + np.spacing(1))
        dice_loss = torch.mean(d)
        return 1 - dice_loss

    def forward(self, model_predict, target, training_masks):
        '''
        :param model_predict: [N * C * H * W]
        :param target: [N * C * H * W]
        :param training_masks: [N * H * W]
        :return: loss: scalar
        '''
        gt_texts = target[:, 0, :, :]
        gt_kernels = target[:, 1: , :, :]

        texts = model_predict[:, 0, :, :]
        selected_masks_texts = self._ohem_batch(texts, gt_texts, training_masks)
        loss_text = self._dice_loss(texts, gt_texts, selected_masks_texts)

        loss_kernels = []
        scores = torch.sigmoid(texts)
        selected_masks_kernels = ((scores > 0.5) & (training_masks > 0.5)).float()
        kernel_num = gt_kernels.shape[1]
        for i in range(kernel_num):
            gt_kernel = gt_kernels[:, i, :, :]
            loss_kernel_i = self._dice_loss(model_predict[:, i + 1, :, :], gt_kernel, selected_masks_kernels)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = sum(loss_kernels) / len(loss_kernels)
        loss = 0.7 * loss_text + 0.3 * loss_kernels
        return loss

