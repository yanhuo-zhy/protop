# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from cProfile import label
import math
import os
import sys
import logging
import pickle
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from torch.nn.modules.loss import _Loss
import tools.utils as utils
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment

def evaluate_accuracy(preds, targets):
    # 预测精度
    targets = targets.astype(int)
    preds = preds.astype(int)

    assert preds.size == targets.size
    D = max(preds.max(), targets.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        w[preds[i], targets[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = preds.size

    total_acc /= total_instances

    return total_acc

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def train_one_epoch(model: torch.nn.Module, criterion: _Loss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    iteration: int,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args=None,
                    set_training_mode=True,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0
    sup_con_crit = SupConLoss()

    for samples, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        # samples = samples.to(device, non_blocking=True)
        samples = torch.cat(samples, dim=0).to(device)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # print("samples.shape:", samples.shape)
            # outputs, auxi_item = model(samples)
            features, auxi_item = model(samples)
            features = torch.nn.functional.normalize(features, dim=-1)

            f1, f2 = [f for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = targets
            loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            # loss = criterion(outputs, targets)

            # if args.use_ppc_loss:
            #     ppc_cov_coe, ppc_mean_coe = args.ppc_cov_coe, args.ppc_mean_coe
            #     total_proto_act, cls_attn_rollout, original_fea_len = auxi_item[2], auxi_item[3], auxi_item[4]
            #     if hasattr(model, 'module'):
            #         ppc_cov_loss, ppc_mean_loss = model.module.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)
            #     else:
            #         ppc_cov_loss, ppc_mean_loss = model.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)

            #     ppc_cov_loss = ppc_cov_coe * ppc_cov_loss
            #     ppc_mean_loss = ppc_mean_coe * ppc_mean_loss
            #     if epoch >= 20:
            #         loss = loss + ppc_cov_loss + ppc_mean_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_img_mask(data_loader, model, device, args):
    logger = logging.getLogger("get mask")
    logger.info("Get mask")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Mask:'

    # switch to evaluation mode
    model.eval()

    all_attn_mask = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            cat_mask = model.get_attn_mask(images)
            all_attn_mask.append(cat_mask.cpu())
    all_attn_mask = torch.cat(all_attn_mask, dim=0) # (num, 2, 14, 14)
    if hasattr(model, 'module'):
        model.module.all_attn_mask = all_attn_mask
    else:
        model.all_attn_mask = all_attn_mask

# @torch.no_grad()
# def evaluate(data_loader, model, device, args):
#     logger = logging.getLogger("validate")
#     logger.info("Start validation")
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     all_token_attn, pred_labels = [], []
#     for images, target, _ in metric_logger.log_every(data_loader, 10, header):
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with torch.cuda.amp.autocast():
#             output, auxi_items = model(images,)
#             loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         _, pred = output.topk(k=1, dim=1)
#         pred_labels.append(pred)

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

#         if args.use_global:
#             global_acc1 = accuracy(auxi_items[2], target)[0]
#             local_acc1 = accuracy(auxi_items[3], target)[0]
#             metric_logger.meters['global_acc1'].update(global_acc1.item(), n=batch_size)
#             metric_logger.meters['local_acc1'].update(local_acc1.item(), n=batch_size)
#         all_token_attn.append(auxi_items[0])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    all_feats, global_feats, local_feats = [], [], []
    targets = []
    
    for images, target, _ in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            features, auxi_items = model(images,)
            
            all_feats.append(features.cpu().numpy())
            global_feats.append(auxi_items[2].cpu().numpy())
            local_feats.append(auxi_items[3].cpu().numpy())
            targets.append(target.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    global_feats = np.concatenate(global_feats, axis=0)
    local_feats = np.concatenate(local_feats, axis=0)
    targets = np.concatenate(targets, axis=0)

    n_clusters = 100
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_feats)

    kmeans_labels = kmeans.labels_
    all_acc = evaluate_accuracy(kmeans_labels, targets)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(global_feats)

    kmeans_labels = kmeans.labels_
    global_acc = evaluate_accuracy(kmeans_labels, targets)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(local_feats)

    kmeans_labels = kmeans.labels_
    local_acc = evaluate_accuracy(kmeans_labels, targets)

    return f"Sup Con- All Acc: {all_acc:.3f}, Global Acc: {global_acc:.3f}, Local Acc: {local_acc:.3f}"   