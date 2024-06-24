"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm,trange

from sklearn.metrics import classification_report, balanced_accuracy_score
# from ..open_clip.tokenizer import HFTokenizer, SimpleTokenizer

def zero_shot_classifier(model, tokenizer, classnames, templates, device,suffix="", amp=True,disable_tqdm=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        current_batch=[]
        for key in tqdm(sorted(classnames.keys(),key=int),disable=disable_tqdm):
            classname=classnames[key]
            # texts=classname
            current_batch.append(classname+suffix)
            if len(current_batch)==2048:
                texts = tokenizer(current_batch).to(device)  # tokenize
                class_embeddings = model.module.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
                current_batch = []
        texts = tokenizer(current_batch).to(device)  # tokenize
        class_embeddings = model.module.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
        # current_batch = []
        zeroshot_weights = torch.cat(zeroshot_weights).T.to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True,all_image_features=None):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    # batch_nr_max=2
    # all_image_features = []
    if not all_image_features:
        all_image_features = []
        with torch.no_grad():
            for images, target in tqdm(dataloader):
                # if  batch_nr_max:
                images = images.to(device)
                target = target.to(device)

                with autocast():
                    # predict
                    image_features = model.module.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100. * image_features @ classifier
                all_image_features.append(image_features.cpu())
                true.append(target.cpu())
                pred.append(logits.float().cpu())

                    # batch_nr_max-=1

    else:
        with torch.no_grad():
            for i, (images, target) in tqdm(enumerate(dataloader)):
                # if  batch_nr_max:
                    image_features = all_image_features[i].to(device)
                    logits = 100. * image_features @ classifier
                    true.append(target.cpu())
                    pred.append(logits.float().cpu())

                    # batch_nr_max-=1

    pred = torch.cat(pred)
    true = torch.cat(true)

    return pred, true, all_image_features

def run_classification2(model,head, classifier, dataloader, device, amp=True,all_image_features=None):
    
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    # batch_nr_max=2
    # all_image_features = []
    if not all_image_features:
        all_image_features = []
        with torch.no_grad():
            for images, target in tqdm(dataloader):
                # if  batch_nr_max:
                images = images.to(device)
                target = target.to(device)

                with autocast():
                    # predict
                    image_features = model.module.encode_image(images)
                    image_features=head(image_features)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100. * image_features @ classifier
                all_image_features.append(image_features.cpu())
                true.append(target.cpu())
                pred.append(logits.float().cpu())

                    # batch_nr_max-=1

    else:
        with torch.no_grad():
            for i, (images, target) in tqdm(enumerate(dataloader)):
                # if  batch_nr_max:
                    image_features = all_image_features[i].to(device)
                    logits = 100. * image_features @ classifier
                    true.append(target.cpu())
                    pred.append(logits.float().cpu())

                    # batch_nr_max-=1

    pred = torch.cat(pred)
    true = torch.cat(true)

    return pred, true, all_image_features
import json
from open_clip import get_input_dtype

def run_classification_transformer(model,head, classifier, dataloader, device,args, amp=True,image_emb=None,text_emb=None,ctx_length=None):
    
    autocast = torch.cuda.amp.autocast if amp else suppress
    input_dtype = get_input_dtype(args.precision)
    pred = []
    true = []
    nb = 0
    print('starting evaluation',flush=True)
    # batch_nr_max=2
    # all_image_features = []
    for i, batch in enumerate(tqdm(dataloader)):
        # print(i,"in daaloader")
        images, texts,paths, attributes, querries, labels = batch
        attributes=[json.loads(a) for a in attributes]

        if args.train_vision:
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            img_emb = model.module.encode_image(images)
        else:
            img_emb=torch.stack([image_emb[i] for i in paths])
        img_emb = img_emb.to(device=device, non_blocking=True)

        batch_inputs=[]
        batch_targets=[]
        if ctx_length == 'random':
            ctx_length = torch.randint(high=len(attributes[0]),size=(1,)).item()
        for i in range(len(paths)):
            if args.train_no_image:
                model_inputs=[]
            else:
                model_inputs=[img_emb[i]]
            if args.include_context and ctx_length != 0:
                if ctx_length is None:
                    for k,v in attributes[i].items():
                        if k==querries[i]:
                            continue
                        model_inputs.append(text_emb[k])
                            # target=text_emb[v]
                        if len(v)>0: curr_text=sum([text_emb[vv] for vv in v])
                        else: curr_text=text_emb["unknown"]
                        model_inputs.append(curr_text)
                        # model_inputs.append(text_emb[v])
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
            
            batch_inputs.append(torch.stack(model_inputs))
            # batch_targets.append(target)
        batch_inputs=torch.stack(batch_inputs).to(device)
        # batch_targets=torch.stack(batch_targets).to(device)

        target = labels.to(device)
     
        with autocast():
            model_out = head(batch_inputs)
            logits = 100. * model_out @ classifier
            true.append(target.cpu())
            pred.append(logits.float().cpu())
            # batch_nr_max-=1

    pred = torch.cat(pred)
    true = torch.cat(true)

    return pred, true

def run_classification_long_context(model, dataloader, tokenizer, classnames, device,args, amp=True,ctx_length=None):
    
    autocast = torch.cuda.amp.autocast if amp else suppress
    input_dtype = get_input_dtype(args.precision)
    pred = []
    true = []
    nb = 0
    print('starting evaluation',flush=True)
    # batch_nr_max=2
    # all_image_features = []
    for i, batch in enumerate(tqdm(dataloader)):
        # print("start batch",i)
        images, texts,paths, attributes, querries, labels = batch
        attributes=[json.loads(a) for a in attributes]

        with torch.no_grad(), autocast():
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            img_emb = model.module.encode_image(images)
            img_emb = img_emb.to(device=device, non_blocking=True)

        batch_inputs=[]
        batch_targets=[]
        batch_logits=[]
        if ctx_length == 'random':
            ctx_length = torch.randint(high=len(attributes[0]),size=(1,)).item()
        for i in range(len(paths)):
            context = []
            if args.include_context and ctx_length != 0:
                if ctx_length is None:
                    for k,v in attributes[i].items():
                        if k==querries[i]:
                            continue
                            # target=text_emb[v]
                        if len(v)>0: curr_text=", ".join(v)
                        else: curr_text="unknown"
                        context.append(curr_text)
                        # model_inputs.append(text_emb[v])
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
                        if len(v)>0: curr_text=", ".join(v)
                        else: curr_text="unknown"
                        context.append(curr_text)
            
            k = querries[i]
            v = attributes[i][k]
            
            context = "; ".join(context)

            # print('building classifier',i)
            classifier = zero_shot_classifier(model,tokenizer,classnames,None,device,suffix=context,disable_tqdm=True)
            # print('done classifier',i)
            with autocast():
                logits = img_emb[i:i+1] @ classifier

            batch_logits.append(logits)
        batch_logits=torch.cat(batch_logits)

        target = labels
     
        # with autocast():
        #     logits = 100. * model_out @ classifier
        true.append(target.cpu())
        pred.append(logits.float().cpu())
        # batch_nr_max-=1

    pred = torch.cat(pred)
    true = torch.cat(true)

    return pred, true




def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in trange(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap

def average_precision_per_class2(scores, targets:torch.Tensor, return_at_1=False):
    ap = torch.zeros(scores.size(1))
    ap1 = torch.zeros(scores.size(1))
    ap5 = torch.zeros(scores.size(1))
    ac5 = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    batch_sz=2048
    for k in trange((scores.size(1)-1)//batch_sz+1):
        # sort scores
        scores_k = scores[:, k*batch_sz:(k+1)*batch_sz]
        targets_k = targets[:, k*batch_sz:(k+1)*batch_sz]
        _, sortind = torch.sort(scores_k, 0, True)
        sortcol = torch.arange(0,sortind.shape[1]).repeat(sortind.shape[0],1)
        truth = targets_k[sortind,sortcol]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg.unsqueeze(1))
        # compute average precision
        ac5[k*batch_sz:(k+1)*batch_sz] = truth[:4,:].any(0).float()
        ap1[k*batch_sz:(k+1)*batch_sz] = precision[0,:]
        try:
            ap5[k*batch_sz:(k+1)*batch_sz] = precision[4,:]
        except:
            ap5[k*batch_sz:(k+1)*batch_sz] = precision[-1,:]
        ap[k*batch_sz:(k+1)*batch_sz] = torch.where(truth.bool(),precision,torch.zeros_like(precision)).sum(0) / torch.clip((truth.sum(0)), min=1)
        # ap[k*batch_sz:(k+1)*batch_sz] = precision[truth.bool()].view_as(truth).sum(0) / torch.clip((truth.sum(0)), min=1)
    if return_at_1:
        return ap, ap1, ap5, ac5
    else:
        return ap

def average_recall_per_sample(scores,targets,return_at_1=False):
    return average_precision_per_class2(scores.T,targets.T,return_at_1=return_at_1)


def recall_at_1_per_class(scores, targets:torch.Tensor):
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    batch_sz=2048
    for k in trange((scores.size(1)-1)//batch_sz+1):
        # sort scores
        scores_k = scores[:, k*batch_sz:(k+1)*batch_sz]
        targets_k = targets[:, k*batch_sz:(k+1)*batch_sz]
        _, sortind = torch.sort(scores_k, 0, True)
        sortcol = torch.arange(0,sortind.shape[1]).repeat(sortind.shape[0],1)
        truth = targets_k[sortind,sortcol]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg.unsqueeze(1))
        # compute average precision
        ap[k*batch_sz:(k+1)*batch_sz] = torch.where(truth.bool(),precision,torch.zeros_like(precision)).sum(0) / torch.clip((truth.sum(0)), min=1)
        # ap[k*batch_sz:(k+1)*batch_sz] = precision[truth.bool()].view_as(truth).sum(0) / torch.clip((truth.sum(0)), min=1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, key, templates, device, amp=True, verbose=False, save_clf=None, load_clfs=[],all_image_features=None,args=None,wandb=None,is_test=False,is_final=False,prefix=""):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
        # torch.save(classifier,"/home/username/open_clip/finetuning/classifiers/"+str(key)+".pt")
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target, all_image_features = run_classification(model, classifier, dataloader, device, amp=amp,all_image_features=all_image_features)
    is_multilabel = (len(target.shape) == 2)

    test_str = "test_" if is_test else ("val_" if not is_final else "val_f_")
    test_str = prefix + test_str

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        
        # random_ap_per_class = average_precision_per_class2(torch.rand_like(logits), target)
        # print("Random expectance for atribute", target.float().mean(),random_ap_per_class.mean().item())
        ap_per_class, ap1_per_class, ap5_per_class, ac5_per_class = average_precision_per_class2(logits, target, return_at_1=True)
        ar_per_sam, ar1_per_sam, ar5_per_sam, ac5_per_sam = average_recall_per_sample(logits, target, return_at_1=True)
        log_data = {
            test_str + key + "_mean_average_precision": ap_per_class.mean().item(),
            test_str + key + "_mean_precision_at_1": ap1_per_class.mean().item(),
            test_str + key + "_mean_precision_at_5": ap5_per_class.mean().item(),
            test_str + key + "_mean_hit_rate_at_5": ac5_per_class.mean().item(),
            test_str + key + "_mean_average_recall": ar_per_sam.mean().item(),
            test_str + key + "_mean_recall_at_1": ar1_per_sam.mean().item(),
            test_str + key + "_mean_recall_at_5": ar5_per_sam.mean().item(),
            test_str + key + "_mean_hit_rate_sample_at_5": ac5_per_sam.mean().item(),
        }
        if args and args.wandb and args.rank==0:
            assert wandb is not None, 'Please install wandb.'
            wandb.log(log_data)
            # wandb.log(log_data)

        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return log_data, all_image_features
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}, all_image_features


def evaluate_long_context(model, dataloader, tokenizer, classnames, key, templates, device, amp=True, verbose=False, save_clf=None, load_clfs=[],all_image_features=None,args=None,wandb=None,is_test=False,is_final=False,prefix=""):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    
    logits, target = run_classification_long_context(model, dataloader, tokenizer, classnames, device, args, amp=amp)
    is_multilabel = (len(target.shape) == 2)

    test_str = "test_" if is_test else ("val_" if not is_final else "val_f_")
    test_str = prefix + test_str

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        
        # random_ap_per_class = average_precision_per_class2(torch.rand_like(logits), target)
        # print("Random expectance for atribute", target.float().mean(),random_ap_per_class.mean().item())
        ap_per_class, ap1_per_class, ap5_per_class, ac5_per_class = average_precision_per_class2(logits, target, return_at_1=True)
        ar_per_sam, ar1_per_sam, ar5_per_sam, ac5_per_sam = average_recall_per_sample(logits, target, return_at_1=True)
        log_data = {
            test_str + key + "_mean_average_precision": ap_per_class.mean().item(),
            test_str + key + "_mean_precision_at_1": ap1_per_class.mean().item(),
            test_str + key + "_mean_precision_at_5": ap5_per_class.mean().item(),
            test_str + key + "_mean_hit_rate_at_5": ac5_per_class.mean().item(),
            test_str + key + "_mean_average_recall": ar_per_sam.mean().item(),
            test_str + key + "_mean_recall_at_1": ar1_per_sam.mean().item(),
            test_str + key + "_mean_recall_at_5": ar5_per_sam.mean().item(),
            test_str + key + "_mean_hit_rate_sample_at_5": ac5_per_sam.mean().item(),
        }
        if args and args.wandb and args.rank==0:
            assert wandb is not None, 'Please install wandb.'
            wandb.log(log_data)
            # wandb.log(log_data)

        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return log_data, all_image_features
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}, all_image_features


def evaluate2(model,head, dataloader, tokenizer, classnames, key, templates, device, amp=True, verbose=False, save_clf=None, load_clfs=[],all_image_features=None,args=None,wandb=None,step=None):
   
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
        # torch.save(classifier,"/home/username/open_clip/finetuning/classifiers/"+str(key)+".pt")
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target, all_image_features = run_classification2(model,head, classifier, dataloader, device, amp=amp,all_image_features=all_image_features)
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        
        # random_ap_per_class = average_precision_per_class2(torch.rand_like(logits), target)
        # print("Random expectance for atribute", target.float().mean(),random_ap_per_class.mean().item())
        ap_per_class = average_precision_per_class2(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")

        log_data = {"mean_average_precision": ap_per_class.mean().item()}  
        if args.wandb and args.rank==0:
            assert wandb is not None, 'Please install wandb.'
            wandb.log(log_data)

        return {"mean_average_precision": ap_per_class.mean().item()}, all_image_features
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}, all_image_features

def evaluate_transformer(model,head, dataloader, tokenizer, classnames, key, templates, device, amp=True, verbose=False, save_clf=None, load_clfs=[],image_emb=None,text_emb=None,args=None,wandb=None,step=None,is_test=False,prefix="",ctx_length=None):
   
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
        # torch.save(classifier,"/home/username/open_clip/finetuning/classifiers/"+str(key)+".pt")
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target = run_classification_transformer(model,head, classifier, dataloader, device,args, amp=amp,image_emb=image_emb,text_emb=text_emb,ctx_length=ctx_length)
    is_multilabel = (len(target.shape) == 2)

    test_str = "test_" if is_test else "val_"
    test_str = prefix + test_str

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        
        # random_ap_per_class = average_precision_per_class2(torch.rand_like(logits), target)
        # print("Random expectance for atribute", target.float().mean(),random_ap_per_class.mean().item())
        # ap_per_class = average_precision_per_class2(logits, target)
        # ap_per_class, ap1_per_class, ap5_per_class, ac5_per_class = average_precision_per_class2(logits, target, return_at_1=True)
        # log_data = {
        #     test_str+"mean_average_precision": ap_per_class.mean().item(),
        #     test_str+"mean_precision_at_1": ap1_per_class.mean().item(),
        #     test_str+"mean_precision_at_5": ap5_per_class.mean().item(),
        #     test_str+"mean_hit_rate_at_5": ac5_per_class.mean().item(),
        # }
        ap_per_class, ap1_per_class, ap5_per_class, ac5_per_class = average_precision_per_class2(logits, target, return_at_1=True)
        ar_per_sam, ar1_per_sam, ar5_per_sam, ac5_per_sam = average_recall_per_sample(logits, target, return_at_1=True)
        log_data = {
            test_str + "mean_average_precision": ap_per_class.mean().item(),
            test_str + "mean_precision_at_1": ap1_per_class.mean().item(),
            test_str + "mean_precision_at_5": ap5_per_class.mean().item(),
            test_str + "mean_hit_rate_at_5": ac5_per_class.mean().item(),
            test_str + "mean_average_recall": ar_per_sam.mean().item(),
            test_str + "mean_recall_at_1": ar1_per_sam.mean().item(),
            test_str + "mean_recall_at_5": ar5_per_sam.mean().item(),
            test_str + "mean_hit_rate_sample_at_5": ac5_per_sam.mean().item(),
        }

        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")

        # log_data = {"mean_average_precision": ap_per_class.mean().item()}  
        if args.wandb and args.rank==0:
            assert wandb is not None, 'Please install wandb.'
            wandb.log(log_data)

        return log_data
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}
    

    