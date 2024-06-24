import json, logging, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

wandb = None

import sys 
sys.path.append("/home/username/open_clip/src")
from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from training.distributed import is_master
from training.zero_shot import zero_shot_eval
from training.precision import get_autocast
from training.train import AverageMeter, unwrap_model, backward, evaluate, postprocess_clip_output, get_clip_metrics, maybe_compute_generative_loss

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.data import get_data

from training.params import parse_args

print(os.environ["CUDA_VISIBLE_DEVICES"])


# args = r'''--save-frequency 1 
#     --zeroshot-frequency 1 
#     --report-to tensorboard 
#     --train-data=/home/username/open_clip/finetuning/train6.csv
#     --val-data=/home/username/open_clip/finetuning/val6.csv
#     --csv-img-key image_path 
#     --csv-caption-key caption 
#     --warmup 10000 
#     --batch-size=128 
#     --lr=1e-3 
#     --wd=0.1 
#     --epochs=30 
#     --workers=8 
#     --csv-separator ,
#     --model RN50'''

args = parse_args(sys.argv[1:])
# print(len(args),args[0])
# args = parse_args(args)

print(args)
# args.distributed=False

# args.model='ViT-B-32'
# args.pretrained='laion2b_s34b_b79k'

from tqdm import tqdm
def train_one_epoch2(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    print("device in train",device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    pbar = tqdm(dataloader, total=len(dataloader))
    for i,batch in enumerate(pbar):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for
            
import re
import glob
import random
import subprocess
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1

def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


from training.distributed import is_master, init_distributed_device, broadcast_object, is_using_distributed
from datetime import datetime
from functools import partial
from torch import optim
from torch.cuda.amp import GradScaler

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync


try:
    import wandb
except ImportError:
    wandb = None
try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

args.distill=False
args.rank=0
args.world_size=1
# args.device="cuda"
args.save_logs=True
args.checkpoint_path="/home/username/open_clip/saved_models/clip1"
# args.save_frequency=100
args.wandb=False


print("is using distributed",is_using_distributed())

# fully initialize distributed device environment
device = init_distributed_device(args)
print(device)
args.device=device
args.distributed=True



if args.copy_codebase:
    copy_codebase(args)


if args.precision == 'fp16':
    logging.warning(
        'It is recommended to use AMP mixed-precision instead of FP16. '
        'FP16 support needs further verification and tuning, especially for train.')

if args.horovod:
    logging.info(
        f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
        f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
elif args.distributed:
    logging.info(
        f'Running in distributed mode with multiple processes. Device: {args.device}.'
        f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
else:
    logging.info(f'Running with a single process. Device {args.device}.')

dist_model = None
args.distill = args.distill_model is not None and args.distill_pretrained is not None
if args.distill:
    #FIXME: support distillation with grad accum.
    assert args.accum_freq == 1
    #FIXME: support distillation with coca.
    assert 'coca' not in args.model.lower()

if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
    # arg is nargs, single (square) image size list -> int
    args.force_image_size = args.force_image_size[0]
random_seed(args.seed, 0)
model_kwargs = {}
if args.siglip:
    model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
    model_kwargs['init_logit_bias'] = -10

model, preprocess_train, preprocess_val = create_model_and_transforms(
    args.model,
    args.pretrained,
    precision=args.precision,
    device=device,
    jit=args.torchscript,
    force_quick_gelu=args.force_quick_gelu,
    force_custom_text=args.force_custom_text,
    force_patch_dropout=args.force_patch_dropout,
    force_image_size=args.force_image_size,
    image_mean=args.image_mean,
    image_std=args.image_std,
    image_interpolation=args.image_interpolation,
    image_resize_mode=args.image_resize_mode,  # only effective for inference
    aug_cfg=args.aug_cfg,
    pretrained_image=args.pretrained_image,
    output_dict=True,
    **model_kwargs,
)


original_model=model


# Setup wandb, tensorboard, checkpoint logging
args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
#args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
if is_master(args):
    print("IN MASTER")
    #args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
    args.tensorboard_path = os.path.join("/home/username/open_clip/saved_models", "tensorboard") if args.tensorboard else ''
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
else:
    args.tensorboard_path = ''
    # args.tensorboard_path= "/home/username/open_clip/saved_models/tensorboard"

import torch
from torch.utils.tensorboard import SummaryWriter

if args.save_logs and args.tensorboard:
    assert tensorboard is not None, "Please install tensorboard."
    writer = tensorboard.SummaryWriter(args.tensorboard_path)
else:
    writer= None


if args.distill:
    # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
    dist_model, _, _ = create_model_and_transforms(
        args.distill_model, 
        args.distill_pretrained,
        device=device,
        precision=args.precision,
        output_dict=True,
    )
if args.use_bnb_linear is not None:
    print('=> using a layer from bitsandbytes.\n'
            '   this is an experimental feature which requires two extra pip installs\n'
            '   pip install bitsandbytes triton'
            '   please make sure to use triton 2.0.0')
    import bitsandbytes as bnb
    from open_clip.utils import replace_linear
    print(f'=> replacing linear layers with {args.use_bnb_linear}')
    linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
    replace_linear(model, linear_replacement_cls)
    model = model.to(device)

random_seed(args.seed, args.rank)

if args.trace:
    model = trace_model(model, batch_size=args.batch_size, device=device)

if args.lock_image:
    # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
    # model.lock_image_tower(
        # unlocked_groups=args.lock_image_unlocked_groups,
        # freeze_bn_stats=args.lock_image_freeze_bn_stats)
    #----------------------------------------------------
    # for k,v in model.named_parameters():
    #     if "visual.transformer.resblocks." in k:
    #         # keep=8
    #         keep=int(args.lock_image_unlocked_groups)
    #         nr=int(k.replace("visual.transformer.resblocks.","").split(".")[0])
    #         if nr>=keep:
    #             v.requires_grad = True
    #         else:
    #             v.requires_grad = False
    # ok=False
    for k,v in model.named_parameters():
        if "visual.transformer.resblocks." in k:
            # keep=8
            keep=int(args.lock_image_unlocked_groups)
            nr=int(k.replace("visual.transformer.resblocks.","").split(".")[0])
            if nr>=keep:
                v.requires_grad = True
                # ok=True
            else:
                v.requires_grad = False
        elif "visual" in k:
            v.requires_grad = False
if args.lock_text:
    # model.lock_text_tower(
    #     unlocked_layers=args.lock_text_unlocked_layers,
    #     freeze_layer_norm=args.lock_text_freeze_layer_norm)
    #----------------------------------------------------
    # for k,v in model.named_parameters():
    #     if "transformer.resblocks." in k and "visual" not in k:
    #             keep=int(args.lock_text_unlocked_layers)
    #             nr=int(k.replace("transformer.resblocks.","").split(".")[0])
    #             if nr>=keep:
    #                 v.requires_grad = True
    #             else:
    #                 v.requires_grad = False
    for k,v in model.named_parameters():
        if "transformer.resblocks." in k and "visual" not in k:
                keep=int(args.lock_text_unlocked_layers)
                nr=int(k.replace("transformer.resblocks.","").split(".")[0])
                if nr>=keep:
                    v.requires_grad = True
                    # ok=True
                else:
                    v.requires_grad = False
        elif "visual" not in k:
            v.requires_grad = False
if args.lock_text and args.lock_image:
    keep1=int(args.lock_image_unlocked_groups)
    keep2=int(args.lock_text_unlocked_layers)
    if keep1==keep2==100:
        print("keep 100")
        tokenizer = get_tokenizer(args.model)
        data = get_data( args,(preprocess_train, preprocess_val),epoch=0, tokenizer=tokenizer, )
        metrics=evaluate(model, data, 0, args, tb_writer=writer, tokenizer=tokenizer)
        print("Metrics:",metrics)
        exit(0)
if not args.lock_text and not args.lock_image:
    print("NO LOCKS")
                

if args.grad_checkpointing:
    model.set_grad_checkpointing()

if args.distributed and not args.horovod:
    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_args = {}
    if args.ddp_static_graph:
        # this doesn't exist in older PyTorch, arg only added if enabled
        ddp_args['static_graph'] = True
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    if args.distill:
        dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)


# create optimizer and scaler
optimizer = None
scaler = None

if args.train_data or args.dataset_type == "synthetic":
    assert not args.trace, 'Cannot train with traced model'

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    if args.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scaler = GradScaler() if args.precision == "amp" else None

start_epoch=0

# args.batch_size=1024
tokenizer = get_tokenizer(args.model)
data = get_data(
    args,
    (preprocess_train, preprocess_val),
    epoch=start_epoch,
    tokenizer=tokenizer,
)

# create scheduler if train
scheduler = None
if 'train' in data and optimizer is not None:
    total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const-cooldown":
        assert args.epochs_cooldown is not None,\
            "Please specify the number of cooldown epochs for this lr schedule."
        cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
        scheduler = const_lr_cooldown(
            optimizer, args.lr, args.warmup, total_steps,
            cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
    else:
        logging.error(
            f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
        exit(1)

import logging
import os
import sys
import numpy as np
import torch


from training.data import get_data
from training.train import train_one_epoch, evaluate
from open_clip import get_tokenizer, create_loss


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

# print("scheduler",scheduler)
loss = create_loss(args)

for epoch in range(start_epoch, args.epochs):
    if is_master(args):
        logging.info(f'Start epoch {epoch}')

    
    train_one_epoch2(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
    completed_epoch = epoch + 1

    if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        metrics=evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)
    print("Metrics:",metrics)
    # Saving checkpoints.
    if args.save_logs:
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": original_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
        

if args.wandb and is_master(args):
    wandb.finish()


