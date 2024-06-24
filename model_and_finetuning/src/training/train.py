import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
    
except ImportError:
    wandb = None
import wandb


from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
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
    for i, batch in enumerate(dataloader):
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

def train_one_epoch_clip_attributes_cosine(model, data, loss, epoch,key, optimizer, scheduler,image_emb, text_emb, args, tb_writer=None):
    torch.autograd.set_detect_anomaly(True)
    # optimizer,optimizer_head=optimizers
    # scheduler,scheduler_head= schedulers
    # loss,loss2=losses
    model.train()

    # print("in train one!!!!!!!")
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = torch.float32
    input_dtype = get_input_dtype(args.precision)
    # head = head.to(device=device, dtype=input_dtype)
  
    data['train_muse'][0].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train_muse'][0].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    print(len(dataloader))
    print(num_batches_per_epoch)
    
    for i, batch in enumerate(tqdm(dataloader,disable=args.rank)):
        # print(i,"in daaloader")
        i_batch = i
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, paths, attributes, querries = batch
        attributes=[json.loads(a) for a in attributes]

        with autocast():
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            # for k,v in model.module.named_parameters():
            #     print(k,v.requires_grad)
            img_emb = model(image=images)["image_features"]
            # print('img_emb',img_emb.flatten()[0],img_emb.requires_grad)
            # img_emb = img_emb.to(device=device, dtype=torch.float32, non_blocking=True)

        batch_targets=[]
        for i in range(len(paths)):
            k = querries[i]
            v = attributes[i][k]
            if len(v)>0: target=sum([text_emb[vv] for vv in v])
            else: target=text_emb["unknown"]

            batch_targets.append(target)
        batch_targets=torch.stack(batch_targets).to(device)

        if args.train_vision: optimizer.zero_grad()

        # print('batch_inputs',batch_inputs.flatten()[:5],batch_inputs.isnan().sum())
        with autocast():
            model_out = img_emb
            if model_out.isnan().sum(): 
                print(f"Epoch {epoch} Batch {i_batch} Found {model_out.isnan().sum()} / {model_out.numel()} NAN in model out",args.rank)
                exit(1)
            # if epoch: print('model out',model_out.flatten()[:5],model_out.isnan().sum())
            # if epoch: print('bat targets',batch_targets.flatten()[:5],batch_targets.isnan().sum())
            losses=loss(model_out,batch_targets,torch.ones(batch_targets.shape[0],device=device))
            if losses.isnan().sum(): 
                print(f"Epoch {epoch} Batch {i_batch} Found {losses.isnan().sum()} NAN in losses",args.rank)
                exit(1)
            # if epoch: print('losses is nan',losses.isnan().sum())
            losses = losses.nan_to_num(0)
            total_loss = losses.mean()
            # print('tot loss',total_loss,total_loss.requires_grad)

        backward(total_loss,None)
        for n,p in model.module.named_parameters():
            if hasattr(p,'grad') and p.grad is not None and p.grad.isnan().sum():
                print(f"Epoch {epoch} Batch {i_batch} Found {p.grad.isnan().sum()} NAN in p.grad {n}",args.rank)
                exit(1)
        optimizer.step()
        for n,p in model.module.named_parameters():
            if hasattr(p,'grad') and p.isnan().sum():
                print(f"Found {p.isnan().sum()} NAN in p {n}",args.rank)
                exit(1)

        log_data = {"lr": optimizer.param_groups[0]["lr"] ,"epoch":epoch}            
        log_data.update({"loss":total_loss})
        if args.wandb and is_master(args):
            assert wandb is not None, 'Please install wandb.'
            # print("log_data",log_data)
            log_data['step'] = step  # for backwards compatibility
            wandb.log(log_data, step=step)


def train_one_epoch_get_embeddings(model, data, loss, epoch, key, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train_classification'][key].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train_classification'][key].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    all_image_features={}
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, paths = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        
        with autocast():
            image_features=model.module.encode_image(images)

            
            for i,k in enumerate(paths):
                all_image_features[k]=image_features[i]

    if is_master(args):
        torch.save(all_image_features,"/home/username/open_clip/finetuning/image_embeddings/image_embbedings_CLIP_"+args.exp_name.split("_")[0]+".pt")
    return all_image_features

from tqdm import tqdm 
def train_one_epoch_get_embeddings_transformer(model, data, loss, epoch, key, optimizer, scaler, scheduler, dist_model, args, tb_writer=None,is_train=True,is_test=False,only_images=False):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data_key = 'train_muse' if is_train else 'test_muse' if is_test else 'val_muse'
    data[data_key][key].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data[data_key][key].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    all_image_features={}
    all_text_features={}
    for i, batch in enumerate(tqdm(dataloader,disable=args.rank)):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images,texts, paths, attributes, querries, *rest = batch
        # print('attributes in batch',len(attributes),[len(a) for a in attributes])
        attributes=[json.loads(a) for a in attributes]

        if not only_images:
            texts_to_process = ['[MASK]']
            for a in attributes:
                # print("aaaaa",a)
                for k,vals in a.items():
                    texts_to_process.append(k)
                    texts_to_process.extend(vals)
            
            # print("LEEEEN before set",len(texts_to_process))
            texts_to_process = set(texts_to_process) - set([k for k,v in all_text_features.items()])
            texts_to_process = list(texts_to_process)
            texts_tokens = data['tokenizer'](texts_to_process)
            texts_tokens = texts_tokens.to(device=device, non_blocking=True)

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        data_time_m.update(time.time() - end)
        # optimizer.zero_grad()

        with torch.no_grad():
            with autocast():
                image_features=model.module.encode_image(images)
                for i,k in enumerate(paths):
                    all_image_features[k]=image_features[i]

                if not only_images:
                    if len(texts_to_process):
                        # print("LEEEEN",len(texts_to_process))
                        text_features=model.module.encode_text(texts_tokens)
                        for i,k in enumerate(texts_to_process):
                            all_text_features[k]=text_features[i]

    all_features = {
        "texts":all_text_features,
        "images":all_image_features,
    }

    val_str = "" if is_train else "test_" if is_test else "val_"
    fine_str = "_F" if args.local_model else ""
    img_str = "_images" if only_images else ""
    if is_master(args):
        torch.save(all_features,f"/home/username/open_clip/finetuning/image_embeddings/image_embbedings_CLIP{fine_str}_transformer_{val_str}"+args.exp_name.split("_")[0]+f"{img_str}.pt")
    return all_features


def train_one_epoch_head(head, data, loss, epoch,key, optimizer, scheduler,classifier,image_emb, args, tb_writer=None):
    # print("in train one!!!!!!!")
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = torch.float32
    # input_dtype = get_input_dtype(args.precision)
    # head = head.to(device=device, dtype=input_dtype)

    head.train()
  
    data['train_classification'][key].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train_classification'][key].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    
    for i, batch in enumerate(dataloader):
        # print(i,"in daaloader")
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, labels, paths = batch
        # print(len(image_emb),'/work/username/data/va_images/O316389.jpg' in image_emb)
        img_emb=torch.stack([image_emb[i] for i in paths])
        img_emb = img_emb.to(device=device, dtype=input_dtype, non_blocking=True)

        optimizer.zero_grad()

       
        with autocast():
            model_out = head(img_emb)
            classifier=classifier.to(device)

            
            labels=labels.to(device)
            target=labels@classifier.T
            losses=loss(model_out,target,torch.ones(target.shape[0],device=device))

            # logits = model_out @ classifier
            # losses = loss(logits, labels)

            total_loss = losses.sum()
            # losses["loss"] = total_loss

        # print('tot loss bef',total_loss)
        backward(total_loss,None)
        # print('tot loss after',total_loss)[]
        # print(epoch,'head grad',head.module.weight.grad.flatten()[:5])
        optimizer.step()

        log_data = {"lr": optimizer.param_groups[0]["lr"] ,"epoch":epoch}            
        log_data.update({"loss":losses})
        if args.wandb and is_master(args):
            assert wandb is not None, 'Please install wandb.'
            # print("log_data",log_data)
            log_data['step'] = step  # for backwards compatibility
            wandb.log(log_data, step=step)


def train_one_epoch_head_transformer(model,head, data, losses, epoch,key, optimizers, schedulers,image_emb, text_emb, args, tb_writer=None):
    torch.autograd.set_detect_anomaly(True)
    optimizer,optimizer_head=optimizers
    scheduler,scheduler_head= schedulers
    loss,loss2=losses
    
    # print("in train one!!!!!!!")
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = torch.float32
    input_dtype = get_input_dtype(args.precision)
    # head = head.to(device=device, dtype=input_dtype)

    head.train()
  
    data['train_muse'][0].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train_muse'][0].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    # print(len(dataloader))
    # print(num_batches_per_epoch)
    
    for i, batch in enumerate(tqdm(dataloader,disable=args.rank)):
        # print(i,"in daaloader")
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)
            scheduler_head(step)

        images, texts, paths, attributes, querries = batch
        attributes=[json.loads(a) for a in attributes]

        if args.train_vision:
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            img_emb = model.module.encode_image(images)
            img_emb = img_emb.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            img_emb=torch.stack([image_emb[i] for i in paths])
        # img_emb = img_emb.to(device=device, dtype=torch.float32, non_blocking=True)

        batch_inputs=[]
        batch_targets=[]
        ctx_length = None
        if args.include_context:   
            if args.context_length_train is not None:
                if args.context_length_train == 'random':
                    ctx_length = torch.randint(high=len(attributes[0]),size=(1,)).item()
                else:
                    ctx_length = args.context_length_train
        for i in range(len(paths)):
            if args.train_no_image:
                model_inputs = []
            else:
                model_inputs=[img_emb[i].to(text_emb["unknown"].device)]
            if args.include_context and ctx_length != 0:
                if ctx_length is None:
                    for k,v in attributes[i].items():
                        if k==querries[i]:
                            continue
                        model_inputs.append(text_emb[k])
                        if len(v)>0: curr_text=sum([text_emb[vv] for vv in v])
                        else: curr_text=text_emb["unknown"]
                        model_inputs.append(curr_text)
                else:
                    ctx_keys = []
                    for k,v in attributes[i].items():
                        if k!=querries[i]:
                            ctx_keys.append(k)
                    indices = torch.rand(len(ctx_keys))
                    vals, indices = indices.topk(ctx_length)
                    ctx_keys = [ctx_keys[idx] for idx in indices.tolist()]
                    for k in ctx_keys:
                        v = attributes[i][k]
                        model_inputs.append(text_emb[k])
                        if len(v)>0: curr_text=sum([text_emb[vv] for vv in v])
                        else: curr_text=text_emb["unknown"]
                        model_inputs.append(curr_text)

            k = querries[i]
            v = attributes[i][k]
            model_inputs.append(text_emb[k])
            model_inputs.append(text_emb["[MASK]"])
            if len(v)>0: target=sum([text_emb[vv] for vv in v])
            else: target=text_emb["unknown"]

            batch_inputs.append(torch.stack(model_inputs))
            batch_targets.append(target)
        batch_inputs=torch.stack(batch_inputs).to(device)
        batch_targets=torch.stack(batch_targets).to(device)

        if args.train_vision: optimizer.zero_grad()
        optimizer_head.zero_grad()

        # print('batch_inputs',batch_inputs.flatten()[:5],batch_inputs.isnan().sum())
        with autocast():
            model_out = head(batch_inputs)
            # print('model out',model_out.flatten()[:5],model_out.isnan().sum())
            # print('bat targets',batch_targets.flatten()[:5],batch_targets.isnan().sum())
            losses=loss(model_out,batch_targets,torch.ones(batch_targets.shape[0],device=device))
            # print('losses is nan',losses.isnan().sum())
            losses = losses.nan_to_num(0)
            total_loss = losses.mean()
            # print('tot loss',total_loss)

        if args.caption_loss:
            with autocast():
                model_out2 = model(images, texts)
                losses2 = loss2(**model_out2, output_dict=True)
                clip_loss= sum(losses2.values())
            total_loss+=-0.05*clip_loss
        
        backward(total_loss,None)
        if args.train_vision: optimizer.step()
        optimizer_head.step()

        log_data = {"lr": optimizer_head.param_groups[0]["lr"] ,"epoch":epoch}            
        log_data.update({"loss":total_loss})
        if args.wandb and is_master(args):
            assert wandb is not None, 'Please install wandb.'
            # print("log_data",log_data)
            log_data['step'] = step  # for backwards compatibility
            wandb.log(log_data, step=step)

import sys
sys.path.append("/home/username/open_clip_original/src/")
import torch.nn.functional as F
from torch import nn
from open_clip.transformer import HeadTransformer

class MyCosineLoss(nn.Module):

    def __init__(self,reduction='none'):
        super().__init__()
        self.reduction=reduction
        # self._backward_hooks = None
        # self._backward_pre_hooks = None

    def forward(self,x:torch.Tensor,y:torch.Tensor,target:torch.Tensor):
        x_norm = x.norm(2,dim=1) + 1e-9
        y_norm = y.norm(2,dim=1) + 1e-9
        cos = (x * y).sum(dim=-1)
        losses = 1 - cos/x_norm/y_norm
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return losses.mean()

def create_head_transformer(arhitecture=None):
    # loss=torch.nn.CosineEmbeddingLoss(reduction='none')
    loss=MyCosineLoss(reduction='none')
    head=HeadTransformer(layers=arhitecture)
    return head,loss

def create_cosine_loss():
    loss=torch.nn.CosineEmbeddingLoss(reduction='none')
    # loss=MyCosineLoss(reduction='none')
    return loss

def create_head(arhitecture=None):
    loss=torch.nn.CosineEmbeddingLoss()

    if arhitecture:
        layers=[nn.Linear(512,512)]
        for i in range(arhitecture-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(512,512))
        head=nn.Sequential(*layers)

        return head,loss
    

    head= nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,512),
        )
    # loss=torch.nn.CosineEmbeddingLoss()

    return head,loss

def create_head_old():
    head=nn.Linear(512,512)
    # loss=torch.nn.CrossEntropyLoss()
    # loss=torch.cosine_similarity
    loss=torch.nn.CosineEmbeddingLoss()

    # head.weight.data=torch.eye(512)+torch.rand_like(head.weight.data)/22.62*0
    # head.bias.data=torch.zeros_like(head.bias.data)+torch.rand_like(head.bias.data)/22.62*0

    return head,loss

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None,is_test=False,prefix=""):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    prefix=prefix + "test" if is_test else prefix + "val"

    if prefix in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data[prefix].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features).to(torch.float32),
                text_features=torch.cat(all_text_features).to(torch.float32),
                logit_scale=logit_scale.to(torch.float32).cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, f"clip_{prefix}_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({f"{prefix}_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {f"{prefix}_" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=None)
        print("I did the log")

    return metrics

from training.zeroshot_classification import evaluate as zeroshot_evaluate
def evaluate_zeroshot(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()



    classnames = data["classes"] if "classes" in data else None
    all_image_features = None
    for key in classnames.keys():
        metrics, all_image_features = zeroshot_evaluate(
            model, 
            data["classes_dtloader"][key].dataloader, 
            tokenizer, 
            classnames[key], templates=None, 
            device="cuda", 
            all_image_features=all_image_features,
        )
        print("metrics ",key,":",metrics)


    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features).to(torch.float32),
                text_features=torch.cat(all_text_features).to(torch.float32),
                logit_scale=logit_scale.to(torch.float32).cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
