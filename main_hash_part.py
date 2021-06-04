
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from utils.config import get_config
from models import build_model
from models.hash_model import DSHNet
from models.hash_loss import DCHLoss
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from utils.feat_extractor import feat_extractor, code_generator
from utils.tools import CalcTopMap
from utils.ret_metric import RetMetric
from models.exchnet_loss import SP_Loss, CH_Loss
from utils.evaluate import mean_average_precision

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    parser.add_argument('--hash_bit', type=int, default=-1, help="Num of hashbit")
    parser.add_argument('--att_size', type=int, default=4, help="Num of att size")
    parser.add_argument('--gamma', type=float, default=20.0, help="Cauchy loss gamma")
    parser.add_argument('--lambd', type=float, default=0.1, help="Cauchy loss lambd")
    parser.add_argument('--lambd_cls', type=float, default=1.0, help="CLS loss lambd")
    parser.add_argument('--pretrained', help='resume from checkpoint')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--valid_rank", type=int, default=0, help='rank for validation gpu')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(args, config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, data_loader_gallery, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = DSHNet(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        crt_cls = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        crt_cls = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        crt_cls = torch.nn.CrossEntropyLoss()
    
    crt_hash = DCHLoss(config)

    crt_sp = SP_Loss()
    crt_ch = CH_Loss()
    
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        # max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        if dist.get_rank() == args.valid_rank:
            mAP = validate(config, model, data_loader_val, data_loader_gallery)
            logger.info(f"mAP of the network on the {len(dataset_val)} query images: {mAP:.3f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, crt_hash, crt_cls, crt_sp, crt_ch, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if dist.get_rank() == args.valid_rank:
            mAP = validate(config, model, data_loader_val, data_loader_gallery)
            logger.info(f"mAP of the network on the {len(dataset_val)} query images: {mAP:.6f}")
            if mAP > max_accuracy:
                save_checkpoint(config, epoch, model_without_ddp, mAP, optimizer, lr_scheduler, logger, best=True)
            max_accuracy = max(max_accuracy, mAP)
            # if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            if (epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
            logger.info(f'Max accuracy: {max_accuracy:.6f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, crt_hash, crt_cls, crt_sp, crt_ch, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    closs_meter = AverageMeter()
    qloss_meter = AverageMeter()
    hash_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    sp_loss_meter = AverageMeter()
    ch_loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        #print("targets.shape: ", targets.shape)
        #print("targets: ", targets)
        labels = torch.eye(config.MODEL.NUM_CLASSES)[targets].cuda(non_blocking=True)
        outputs, preds, sp_v, ch_v = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            hash_loss, c_loss, q_loss = crt_hash(outputs, labels)
            cls_loss = crt_cls(preds, targets)
            sp_loss = crt_sp(sp_v)
            ch_loss = crt_sp(ch_v)
            loss = hash_loss + config.HASH.LAMBD_CLS * cls_loss + \
                 config.HASH.LAMBD_SP * sp_loss + config.HASH.LAMBD_CH * ch_loss
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            c_loss = c_loss / config.TRAIN.ACCUMULATION_STEPS
            q_loss = q_loss / config.TRAIN.ACCUMULATION_STEPS
            cls_loss = cls_loss / config.TRAIN.ACCUMULATION_STEPS
            hash_loss = hash_loss / config.TRAIN.ACCUMULATION_STEPS
            sp_loss = sp_loss / config.TRAIN.ACCUMULATION_STEPS
            ch_loss = ch_loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            hash_loss, c_loss, q_loss = crt_hash(outputs, labels)
            cls_loss = crt_cls(preds, targets)
            sp_loss = crt_sp(sp_v)
            ch_loss = crt_sp(ch_v)
            loss = hash_loss + config.HASH.LAMBD_CLS * cls_loss + \
                 config.HASH.LAMBD_SP * sp_loss + config.HASH.LAMBD_CH * ch_loss
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        closs_meter.update(c_loss.item(), targets.size(0))
        qloss_meter.update(q_loss.item(), targets.size(0))
        hash_loss_meter.update(hash_loss.item(), targets.size(0))
        cls_loss_meter.update(cls_loss.item(), targets.size(0))
        sp_loss_meter.update(sp_loss.item(), targets.size(0))
        ch_loss_meter.update(ch_loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\n'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'closs {closs_meter.val:.4f} ({closs_meter.avg:.4f})\t'
                f'qloss {qloss_meter.val:.4f} ({qloss_meter.avg:.4f})\t'
                f'hash_loss {hash_loss_meter.val:.4f} ({hash_loss_meter.avg:.4f})\n'
                f'cls_loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t'
                f'sp_loss {sp_loss_meter.val:.4f} ({sp_loss_meter.avg:.4f})\t'
                f'ch_loss {ch_loss_meter.val:.4f} ({ch_loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

best_mapr = 0
best_iter = -1
@torch.no_grad()
def validate(config, model, test_loader, database_loader):
    global best_mapr
    global best_iter
    crt = torch.nn.CrossEntropyLoss()
    model.eval()
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    test_codes, test_labels = code_generator(model, test_loader, logger)
    test_labels_onehot = torch.eye(config.MODEL.NUM_CLESSES)[test_labels].cuda(non_blocking=True)
    gallery_codes, gallery_labels = code_generator(model, database_loader, logger)
    gallery_labels_onehot = torch.eye(config.MODEL.NUM_CLESSES)[gallery_labels].cuda(non_blocking=True)
    # mAP = CalcTopMap(gallery_codes, test_codes, gallery_labels_onehot.numpy(),
    #                  test_labels_onehot.numpy(), 10000)
    mAP = mean_average_precision(test_codes, gallery_codes, test_labels_onehot, gallery_labels_onehot, -1)
    # pr_range = [10, 20, 40, 80]
    # codes = [gallery_codes, test_codes]
    # labels = [gallery_labels.numpy(), test_labels.numpy()]
    if mAP > best_mapr:
        best_mapr = mAP
    logger.info(f' mAP {mAP:.6f} best_mAP {best_mapr:.6f}')
    #r_k_func = RetMetric(codes, labels, hamming_dis=True)
    #r_k_list = []
    #logger.info(f'Recall@K\t')
    #for i, k in enumerate(pr_range):
    #    v = r_k_func.recall_k(k)
    #    logger.info(f'{k}: {v:.6f}')
    return mAP


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), show_rank=args.valid_rank, name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(args, config)
