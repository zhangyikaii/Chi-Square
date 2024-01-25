import datetime
import logging
import time
from pathlib import Path
import os

import numpy as np
from sklearn.utils import indices_to_mask
import torch

from debias.datasets.biased_mnist import get_dataloader
from debias.utils.logging import set_logging
from debias.utils.utils import (AverageMeter, set_seed, PrepareFunc, parse_option, pairwise_metric, categorical_accuracy, MultiStepScheduler, get_conflict_feature, get_mixed_prototypes, ValHandle, DictTorchAdder, save_pickle)
from debias.utils.logger import Logger



def indices2boolindices(indices, length):
    ans = torch.zeros(length).bool()
    ans[indices] = True
    return ans

def boolindices2indices(indices):
    return torch.arange(len(indices))[indices]

def local_sample(feat, y, num):
    indices = torch.randperm(feat.shape[0])[: num]
    return feat[indices], y[indices]

def train(train_loader, model, criterion, optimizer, epoch, opt, weight_handle, logger):
    model.train()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    conflict_data = train_loader.dataset.conflict_data.cuda()
    conflict_targets = train_loader.dataset.conflict_targets.cuda()
    conflict_bool_indices = train_loader.dataset.conflict_bool_indices
    # conflict_biased_targets = train_loader.dataset.conflict_biased_targets.cuda()

    len_train_iter = len(train_iter)

    cur_weight = weight_handle.get(epoch)
    for idx, (images, labels, biases, indices) in enumerate(train_iter):
        conflict_embs = get_conflict_feature(model, conflict_data, conflict_targets, opt.feat_norm, int((1 - opt.corr) * len(train_loader.dataset)), is_return_feat=True)

        c_indices = boolindices2indices(indices2boolindices(indices, len(conflict_bool_indices)) & conflict_bool_indices)
        indices_bool_c = torch.zeros_like(indices).bool()
        if c_indices.shape[0] != 0:
            for i in c_indices:
                indices_bool_c |= (i == indices)

        model.train()

        cur_logger_step = epoch * len_train_iter + idx
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        _, feat = model(images, is_norm=opt.feat_norm)


        prototypes_to_aligned_batch = get_mixed_prototypes(labels, conflict_targets, feat, conflict_embs, opt.protonet_sampled_num, opt.protonet_conflict_rate)
        ab_conflict_num = int(len(indices[~indices_bool_c]) / opt.batch_ratio * (1 - opt.batch_ratio))
        if ab_conflict_num < len(indices[indices_bool_c]):
            tmp_c, tmp_t = local_sample(feat[indices_bool_c], labels[indices_bool_c], len(indices[indices_bool_c]) - ab_conflict_num)
            cur_aligned_batch = torch.cat([feat[~indices_bool_c], tmp_c])
            cur_aligned_batch_target = torch.cat([labels[~indices_bool_c], tmp_t])
        else:
            tmp_c, tmp_t = local_sample(conflict_embs, conflict_targets, ab_conflict_num - len(indices[indices_bool_c]))
            cur_aligned_batch = torch.cat([feat, tmp_c])
            cur_aligned_batch_target = torch.cat([labels, tmp_t])

        if cur_weight != 0:
            logits_to_aligned_batch = pairwise_metric(
                x=cur_aligned_batch,
                y=prototypes_to_aligned_batch,
                matching_fn=opt.metric,
                temperature=opt.temperature,
                is_distance=False
                )
            to_aligned_loss = criterion(logits_to_aligned_batch, cur_aligned_batch_target)
            loss = to_aligned_loss

        if cur_weight != 1:
            prototypes_to_conflict_batch = get_mixed_prototypes(labels, conflict_targets, feat, conflict_embs, opt.protonet_sampled_num, 1 - opt.protonet_conflict_rate)
            cb_conflict = torch.cat([feat[indices_bool_c], conflict_embs])
            cb_conflict_target = torch.cat([labels[indices_bool_c], conflict_targets])
            cb_aligned = feat[~indices_bool_c]
            cb_conflict_num = min(len(cb_conflict), int(opt.batch_ratio * len(cur_aligned_batch)))
            cb_aligned_num = int(cb_conflict_num / opt.batch_ratio * (1 - opt.batch_ratio))

            tmp_cb_c, tmp_cb_ct = local_sample(cb_conflict, cb_conflict_target, cb_conflict_num)
            tmp_cb_a, tmp_cb_at = local_sample(cb_aligned, labels[~indices_bool_c], cb_aligned_num)
            cur_conflict_batch, cur_conflict_batch_target = torch.cat([tmp_cb_c, tmp_cb_a]), torch.cat([tmp_cb_ct, tmp_cb_at])

            logits_to_conflict_batch = pairwise_metric(
                x=cur_conflict_batch,
                y=prototypes_to_conflict_batch,
                matching_fn=opt.metric,
                temperature=opt.temperature,
                is_distance=False
                )
            to_conflict_loss = criterion(logits_to_conflict_batch, cur_conflict_batch_target)


        if cur_weight == 0:
            loss = to_conflict_loss
        elif cur_weight == 1:
            loss = to_aligned_loss
        else:
            loss = to_aligned_loss * cur_weight + to_conflict_loss * (1 - cur_weight)

        logger.add_scalar('Step_Loss', loss.item(), cur_logger_step)

        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg


def main():
    cur_model_name = 'chi_square'
    cur_save_path = 'your/save/path'
    opt = parse_option()

    if opt.time_str == '':
        opt.time_str = datetime.datetime.now().strftime(
            '%m%d-%H-%M-%S-%f')[:-3]
    exp_name = f'{opt.time_str}-lr-{opt.lr}-optmz-{opt.optimizer}-sche-{opt.lr_scheduler}-temp-{opt.temperature}'
    save_path = Path(
        os.path.join(cur_save_path, f'{cur_model_name}-{opt.dataset}-{opt.corr}-{opt.severity}/{exp_name}')
    )
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    train_loader = get_dataloader(
        dataset_name=opt.dataset,
        batch_size=opt.bs,
        data_label_correlation=opt.corr,
        severity=opt.severity,
        split='train',
        train_weight_sampler=opt.train_weight_sampler,
        train_weight_clip_ratio=opt.train_weight_clip_ratio,
        use_selected=opt.use_selected,
        stage1_path=opt.stage1_path,
        conflict_used_rate=opt.conflict_used_rate)

    logger = Logger(opt, save_path)
    val_loaders = {}
    val_loaders['test'] = get_dataloader(dataset_name=opt.dataset,
                                         batch_size=256,
                                         data_label_correlation=opt.corr,
                                         severity=1,
                                         split='valid')

    prepare_handle = PrepareFunc(opt)
    model = prepare_handle.prepare_model()
    criterion = prepare_handle.prepare_loss_fn()
    optimizer, scheduler = prepare_handle.prepare_optimizer(model)

    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    loss_weight_handle = MultiStepScheduler(opt.weight, opt.weight_increase,
                                            opt.epochs, opt.points)

    val_handle = ValHandle(cur_model_name, val_loaders['test'], opt)

    if opt.only_do_test:
        logging.info(f'Skip training...')
        # TODO
        return

    for epoch in range(1, opt.epochs + 1):
        logging.info(
            f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}'
        )
        loss = train(train_loader, model, criterion, optimizer, epoch, opt,
                     loss_weight_handle, logger)

        logger.add_scalar('Epoch_Loss', loss, epoch)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss}')

        scheduler.step()

        val_handle.val(
            epoch,
            model,
            logging,
            optimizer,
            save_path,
            skip_attrwise_acc=True if opt.dataset == 'NICO' else False,
            is_val_proto=True,
            train_loader=train_loader,
            only_save_best_model=True
            if opt.dataset in ['CorruptedCIFAR10-Type0', 'NICO'] else False)

        if opt.train_acc_early_stop and epoch - val_handle.best_u(
        )['epoch'] > int(opt.epochs * 0.5):
            logging.info('Bad Training Error.')
            val_handle.result_log(
                False,
                log_str=
                f'{opt.weight},{opt.lr},{opt.corr},{opt.auxiliary_weight},{opt.vanilla_train_weight_sampler_from_file},{opt.train_weight_clip_ratio},{opt.stage1_path},{opt.notes},{opt.train_weight_sampler},{opt.conflict_used_rate},{opt.seed}'
            )
            raise Exception('Bad Training Error.')

    val_handle.result_log(
        True,
        log_str=
        f'{opt.weight},{opt.lr},{opt.corr},{opt.auxiliary_weight},{opt.vanilla_train_weight_sampler_from_file},{opt.train_weight_clip_ratio},{opt.stage1_path},{opt.notes},{opt.train_weight_sampler},{opt.conflict_used_rate},{opt.seed}'
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')



if __name__ == '__main__':
    main()
