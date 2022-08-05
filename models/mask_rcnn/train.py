import argparse
import os
import sys
import pickle
import resource
import logging
from collections import defaultdict
import torchvision.models.segmentation
import numpy as np
import yaml
import torch
from PIL import Image
import time
from tqdm import tqdm
import cv2
from config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from roidb import combined_roidb_for_training
from loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from logging_ import setup_logging
from timer import Timer
import pycocotools.mask as mask_utils
from torch.utils.tensorboard import SummaryWriter
from utils import *

torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')
    # Optimization
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)
    # Epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')
    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')

    # dataset options
    parser.add_argument(
        '--clevr_comp_cat',
        help='Use compositional categories for clevr dataset',
        default=1, type=int)

    return parser.parse_args()


def save_ckpt(checkpoint_path, args, step, train_size, model, optimizer):

    save_name = os.path.join(checkpoint_path, 'model_step{}.pth'.format(step))
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)


def main():

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if 'clevr' in args.dataset:
        cfg.TRAIN.DATASETS = ('clevr_mini',)
        cfg.TRAIN.SCALES = (320, )
        if args.clevr_comp_cat:
            cfg.MODEL.NUM_CLASSES = 49
            cfg.CLEVR.COMP_CAT = True
        else:
            cfg.MODEL.NUM_CLASSES = 4
            cfg.CLEVR.COMP_CAT = False
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    effective_batch_size = args.batch_size

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

    ### Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

    ### Adjust solver steps
    step_scale = original_batch_size / effective_batch_size
    old_solver_steps = cfg.SOLVER.STEPS
    old_max_iter = cfg.SOLVER.MAX_ITER
    cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
    print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
                                                  old_max_iter, cfg.SOLVER.MAX_ITER))

    # Scale FPN rpn_proposals collect size (post_nms_topN) in `collect` function
    # of `collect_and_distribute_fpn_rpn_proposals.py`
    #
    # post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
        cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
        print('Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
              '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(cfg.FPN.RPN_COLLECT_SCALE))

    cfg.DATA_LOADER.NUM_THREADS = 4
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma
    assert_and_infer_cfg()

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()
    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    roidb_size = len(roidb)
    logger.info('{:d} roidb entries'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    # Effective training sample size for one epoch
    train_size = roidb_size // args.batch_size * args.batch_size

    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=args.batch_size,
        drop_last=True
    )
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=0, #cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch
    )

    dataiterator = iter(dataloader)

    maskRCNN = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = maskRCNN.roi_heads.box_predictor.cls_score.in_features
    maskRCNN.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=cfg.MODEL.NUM_CLASSES)

    print(f"Treinable Parameters: {sum(p.numel() for p in maskRCNN.parameters() if p.requires_grad)}")

    maskRCNN.cuda()

    # gn_param_nameset = set()
    # for name, module in maskRCNN.named_modules():
    #     if isinstance(module, nn.GroupNorm):
    #         gn_param_nameset.add(name+'.weight')
    #         gn_param_nameset.add(name+'.bias')
    #
    # gn_params = []
    # gn_param_names = []
    # bias_params = []
    # bias_param_names = []
    # nonbias_params = []
    # nonbias_param_names = []
    # nograd_param_names = []
    # for key, value in maskRCNN.named_parameters():
    #     if value.requires_grad:
    #         if 'bias' in key:
    #             bias_params.append(value)
    #             bias_param_names.append(key)
    #         elif key in gn_param_nameset:
    #             gn_params.append(value)
    #             gn_param_names.append(key)
    #         else:
    #             nonbias_params.append(value)
    #             nonbias_param_names.append(key)
    #     else:
    #         nograd_param_names.append(key)
    # assert (gn_param_nameset - set(nograd_param_names) - set(bias_param_names)) == set(gn_param_names)
    #
    # # Learning rate of 0 is a dummy value to be set properly at the start of training
    # params = [
    #     {'params': nonbias_params,
    #      'lr': 0,
    #      'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
    #     {'params': bias_params,
    #      'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
    #      'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
    #     {'params': gn_params,
    #      'lr': 0,
    #      'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    # ]
    # # names of paramerters for each paramter
    # param_names = [nonbias_param_names, bias_param_names, gn_param_names]

    #if cfg.SOLVER.TYPE == "SGD":
    #    optimizer = torch.optim.SGD(maskRCNN.parameters(), momentum=cfg.SOLVER.MOMENTUM, lr=1e-3)
    #elif cfg.SOLVER.TYPE == "Adam":
    optimizer = torch.optim.Adam(maskRCNN.parameters(), lr=1e-5)

    ### Load checkpoint
    # if args.load_ckpt:
    #     load_name = args.load_ckpt
    #     logging.info("loading checkpoint %s", load_name)
    #     checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    #     net_utils.load_ckpt(maskRCNN, checkpoint['model'])
    #     if args.resume:
    #         args.start_step = checkpoint['step'] + 1
    #         if 'train_size' in checkpoint:  # For backward compatibility
    #             if checkpoint['train_size'] != train_size:
    #                 print('train_size value: %d different from the one in checkpoint: %d'
    #                       % (train_size, checkpoint['train_size']))
    #
    #         # reorder the params in optimizer checkpoint's params_groups if needed
    #         # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)
    #
    #         # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
    #         # However it's fixed on master.
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         # misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
    #     del checkpoint
    #     torch.cuda.empty_cache()

    # lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    # Set the folder time for each execution
    folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

    # Set the model name
    args.run_name = f"exec_{folder_time}"
    args.cfg_filename = os.path.basename(args.cfg_file)

    output_path = os.path.join(cfg.OUTPUT_DIR, 'mask_rcnn', args.run_name)
    checkpoint_path = os.path.join(output_path, 'checkpoint')

    # Create the folders
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Save configuration
    blob = {'cfg': yaml.dump(cfg), 'args': args}
    with open(os.path.join(output_path, 'config_and_args.pkl'), 'wb') as f:
        pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

    writer = SummaryWriter(log_dir=output_path)

    # Set index for decay steps
    # decay_steps_ind = None
    # for i in range(1, len(cfg.SOLVER.STEPS)):
    #     if cfg.SOLVER.STEPS[i] >= args.start_step:
    #         decay_steps_ind = i
    #         break
    # if decay_steps_ind is None:
    #     decay_steps_ind = len(cfg.SOLVER.STEPS)
    #
    # training_stats = TrainingStats(
    #     args,
    #     args.disp_interval,
    #     tblogger if args.use_tfboard and not args.no_save else None)
    is_plotting = False
    logger.info('Training starts !')
    batch_time = AverageMeter()

    tic = time.time()
    with tqdm(total=cfg.SOLVER.MAX_ITER) as pbar:

        for step in range(cfg.SOLVER.MAX_ITER):

            # Warm up
            # if step < cfg.SOLVER.WARM_UP_ITERS:
            #     method = cfg.SOLVER.WARM_UP_METHOD
            #     if method == 'constant':
            #         warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
            #     elif method == 'linear':
            #         alpha = step / cfg.SOLVER.WARM_UP_ITERS
            #         warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
            #     else:
            #         raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
            #     lr_new = cfg.SOLVER.BASE_LR * warmup_factor
            #     net_utils.update_learning_rate(optimizer, lr, lr_new)
            #     lr = optimizer.param_groups[0]['lr']
            #     assert lr == lr_new
            # elif step == cfg.SOLVER.WARM_UP_ITERS:
            #     net_utils.update_learning_rate(optimizer, lr, cfg.SOLVER.BASE_LR)
            #     lr = optimizer.param_groups[0]['lr']
            #     assert lr == cfg.SOLVER.BASE_LR
            #
            # # Learning rate decay
            # if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
            #         step == cfg.SOLVER.STEPS[decay_steps_ind]:
            #     logger.info('Decay the learning on step %d', step)
            #     lr_new = lr * cfg.SOLVER.GAMMA
            #     net_utils.update_learning_rate(optimizer, lr, lr_new)
            #     lr = optimizer.param_groups[0]['lr']
            #     assert lr == lr_new
            #     decay_steps_ind += 1

            optimizer.zero_grad()

            try:
                input_data = next(dataiterator)
            except StopIteration:
                dataiterator = iter(dataloader)
                input_data = next(dataiterator)

            images = input_data['data'][0].cuda()
            roidb = input_data['roidb'][0]

            targets = []

            for b in roidb:
                b = b[0]
                boxes = torch.tensor(b['boxes']).cuda()
                labels = torch.tensor(b['gt_classes'], dtype=torch.int64).cuda()
                masks = torch.as_tensor([mask_utils.decode(x) for x in b['segms']]).cuda()
                targets.append({'boxes': boxes, 'labels': labels, 'masks': masks})

            maskRCNN.train()
            losses = maskRCNN(images, targets)

            loss_classifier = losses['loss_classifier']
            loss_box_reg = losses['loss_box_reg']
            loss_mask = losses['loss_mask']
            loss_objectness = losses['loss_objectness']
            loss_rpn_box_reg = losses['loss_rpn_box_reg']

            losses = sum(loss for loss in losses.values())
            losses.backward()
            optimizer.step()

            # Save tensorflow board
            writer.add_scalar('loss_classifier', loss_classifier.item(), step)
            writer.add_scalar('loss_box_reg', loss_box_reg.item(), step)
            writer.add_scalar('loss_mask', loss_mask.item(), step)
            writer.add_scalar('loss_objectness', loss_objectness.item(), step)
            writer.add_scalar('loss_rpn_box_reg', loss_rpn_box_reg.item(), step)

            # Measure elapsed time
            toc = time.time()
            batch_time.update(toc - tic)

            # Set the var description
            pbar.set_description(("{:.1f}s - loss_classifier: {:.6f}, loss_box_reg: {:.6f}, loss_mask: {:.6f}, "
                                  "loss_objectness: {:.6f}, loss_rpn_box_reg: {:.6f}".format(
                (toc - tic), loss_classifier.item(), loss_box_reg.item(), loss_mask.item(),
                loss_objectness.item(), loss_rpn_box_reg.item())))

            # Update the bar
            pbar.update(1)

            if is_plotting:
                plot_input(images, targets)

            if step % 100 == 0:

                # Save Checkpoint
                save_ckpt(checkpoint_path, args, step, train_size, maskRCNN, optimizer)

                final_image = evaluate(maskRCNN, images[0])
                grid = torchvision.utils.make_grid(final_image)
                writer.add_image(f"images", grid, step)

    # Save last checkpoint
    save_ckpt(checkpoint_path, args, step, train_size, maskRCNN, optimizer)

    writer.close()


def plot_input(images, targets, id_data=0):

    image = np.array(images[id_data].cpu())

    image = np.transpose(image, (1, 2, 0))
    image += cfg.PIXEL_MEANS
    image = image.astype(np.uint8, copy=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = targets[id_data]['boxes']
    masks = targets[id_data]['masks']

    image_mask = image.copy()
    final_mask = np.zeros_like(masks[0].cpu())

    for b, m in zip(boxes, masks):
        m = np.array(m.cpu())
        m = m.astype(np.uint8, copy=False)
        final_mask = final_mask | m

        b = np.array(b.cpu())
        b = b.astype(np.uint, copy=False)
        x1, y1, x2, y2 = b
        cv2.rectangle(image, (x1, y2), (x2, y1), (0, 0, 255), 2)

    # Apply final mask
    image_mask = cv2.bitwise_and(image_mask, image_mask, mask=final_mask)

    cv2.imshow('image_bbox', image)
    cv2.imshow('image_mask', image_mask)
    cv2.waitKey(0)


def evaluate(maskRCNN, image):

    maskRCNN.eval()
    prediction = maskRCNN(image.unsqueeze(dim=0))
    prediction = prediction[0]

    image = np.array(image.cpu())

    image = np.transpose(image, (1, 2, 0))
    image += cfg.PIXEL_MEANS
    image = image.astype(np.uint8, copy=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = prediction['boxes'].detach()
    masks = prediction['masks'].detach()

    image_mask = image.copy()
    image_bbox = image.copy()

    if len(masks) > 0 and len(boxes) > 0:

        final_mask = np.zeros_like(masks[0][0].cpu(), dtype=np.uint)

        for b, m in zip(boxes, masks):
            m = np.array(m[0].cpu())
            m[m >= 0.5] = 1
            m[m < 0.5] = 0
            m = m.astype(np.uint, copy=False)
            final_mask = final_mask | m

            b = np.array(b.cpu())
            b = b.astype(np.uint, copy=False)
            x1, y1, x2, y2 = b
            cv2.rectangle(image_bbox, (x1, y2), (x2, y1), (0, 0, 255), 2)

        # Apply final mask
        final_mask = final_mask.astype(np.uint8, copy=False)
        image_mask = cv2.bitwise_and(image_mask, image_mask, mask=final_mask)

    image_bbox = torch.as_tensor(image_bbox.astype(np.uint8, copy=False))
    image_bbox = torch.transpose(image_bbox, 0, 2)
    image_bbox = torch.transpose(image_bbox, 1, 2)
    image_mask = torch.as_tensor(image_mask.astype(np.uint8, copy=False))
    image_mask = torch.transpose(image_mask, 0, 2)
    image_mask = torch.transpose(image_mask, 1, 2)

    final_image = torch.stack([image_bbox, image_mask])

    return final_image


if __name__ == '__main__':
    main()

