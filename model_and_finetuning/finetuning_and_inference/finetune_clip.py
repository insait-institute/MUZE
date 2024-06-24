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


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    # if args.distill:
    #     dist_model.eval()
    
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))


    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    ####################################################################
    pbar = tqdm(dataloader, total=len(dataloader))
    for i,batch in enumerate(pbar):
    # for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

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

        # if args.grad_clip_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        optimizer.step() 
        ############################################################################################



        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
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
    
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
        
    # end for
            
from training.params import parse_args

args = r'''--save-frequency 1 
    --zeroshot-frequency 1 
    --report-to tensorboard 
    --train-data=/home/username/open_clip/finetuning/train6.csv
    --val-data=/home/username/open_clip/finetuning/val6.csv
    --csv-img-key image_path 
    --csv-caption-key caption 
    --warmup 10000 
    --batch-size=1024 
    --lr=1e-3 
    --wd=0.1 
    --epochs=7
    --workers=8 
    --csv-separator ,
    --model RN50'''

# args = parse_args(sys.argv[1:])

args = args.split()
print(len(args),args[0])
args = parse_args(args)

print(args)
args.distributed=False

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.data import get_data

# args.model='ViT-B-32'
# args.pretrained='laion2b_s34b_b79k'
	
# args.model='ViT-L-14-quickgelu'
# args.pretrained='dfn2b'
print(args)

model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device="cuda",
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
        # **model_kwargs,
    )

start_epoch=0

# args.batch_size=1024
tokenizer = get_tokenizer(args.model)
data = get_data(
    args,
    (preprocess_train, preprocess_val),
    epoch=start_epoch,
    tokenizer=tokenizer,
)


exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
include = lambda n, p: not exclude(n, p)
named_parameters = list(model.named_parameters())
gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

optimizer = torch.optim.AdamW(
    [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.wd},
    ],
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
)


from training.scheduler import const_lr #,cosine_lr, const_lr_cooldown
from tqdm import tqdm 

args.distill=False
args.rank=0
args.world_size=1
args.device="cuda"
args.save_logs=True
args.checkpoint_path="/home/username/open_clip/saved_models/clip1"
# args.save_frequency=1
args.wandb=False


scaler=None
loss=create_loss(args)
total_steps = (data["train"].dataloader.num_batches) * args.epochs
scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
dist_model=None
LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
original_model=model


for epoch in range(start_epoch, args.epochs):
    if is_master(args):
        logging.info(f'Start epoch {epoch}')

    
    train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None)
    completed_epoch = epoch + 1

    if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        metrics=evaluate(model, data, completed_epoch, args, tb_writer=None, tokenizer=tokenizer)
    print("Metrics:",metrics)
    # print(type(metrics))


    ## Saving checkpoints.
    if args.save_logs:
        args.name="ep: "+str(epoch)
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": original_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # if scaler is not None:
        #     checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}_itr_{metrics['image_to_text_mean_rank']}.pt"),
            )
        if args.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

        # if args.save_most_recent:
        #     # try not to corrupt the latest checkpoint if save fails
        #     tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
        #     latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
        #     torch.save(checkpoint_dict, tmp_save_path)
        #     os.replace(tmp_save_path, latest_save_path)
    

