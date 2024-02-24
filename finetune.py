### Adapted from mlfoundations/wise-ft repo

import os
import copy
import time
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from random import seed
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from args import parse_arguments
from open_clip import tokenize
from models.modeling import ImageClassifier
from models.utils import cosine_lr, torch_load, LabelSmoothing
from utils import update_confusion_matrix, acc_from_confusion_matrix
from dataloader import get_img_dataloader, get_video_dataloader, get_tufts_dataloader

# seed(12)
# torch.random.manual_seed(12)
# np.random.seed(12)

def eval(model, dataloader, loss_fn, conf_AS):
    losses = []
    preds = []
    labels = []
    ## count = 0
    with torch.set_grad_enabled(False):
        
        for cine, _, label_AS, _ in tqdm(dataloader):
            ## count+=1
            
            cine = cine.squeeze(0)
            cine = cine.cuda()
            label_AS = label_AS.cuda()

            pred = model(cine) #[f, 4] 
            pred = pred.mean(dim=0) #[f, 4]
            preds.append(pred.tolist())
            labels.append(int(label_AS))
            loss = loss_fn(pred.unsqueeze(0), label_AS)
            losses += [loss]

    mean_loss = torch.mean(torch.stack(losses)).item()

    ## calculate confusion matrix + accuracy
    preds = torch.tensor(preds) #[n, 4]
    labels = torch.tensor(labels) 

    prob = F.softmax(preds, dim=-1) #[n, 4]
    _, argm = torch.max(prob, dim=-1) #[n]
    conf_AS = update_confusion_matrix(conf_AS, labels.cpu(), argm.cpu())
    #acc_AS = acc_from_confusion_matrix(conf_AS)
    acc_AS = balanced_accuracy_score(labels.cpu(), argm.cpu())
    print(conf_AS)
    
    return mean_loss, acc_AS

def test_comprehensive_private(args, model, loader, run_name):
    """Logs the network outputs in dataloader
    computes per-patient preds and outputs result to a DataFrame"""
    # Choose which model to evaluate.
    model.eval()
    model = model.cuda()

    with torch.set_grad_enabled(False):
        patient, target_AS_arr, pred_AS_arr, logits = [], [], [], []
        
        for cine, _, label_AS, echo_id in tqdm(loader):
            # collect the label info
            target_AS_arr.append(int(label_AS[0]))
            patient.append(echo_id)
            # Transfer data from CPU to GPU.
            cine = cine.squeeze(0) #[f, c, h, w]
            cine = cine.cuda()
            label_AS = label_AS.cuda()
            echo_id = echo_id.cuda()
            #view.append(data_info['view'][0])
            
            pred = model(cine) #[f, 4] 
            pred = pred.mean(dim=0) #[f, 4]

            prob = F.softmax(pred, dim=-1) #[n, 4]
            _, argm = torch.max(prob, dim=-1) #[n]
            # collect the model prediction info
            logits.append(prob.cpu().numpy())
            pred_AS_arr.append(argm.cpu().numpy())

                    
        # compile the information into a dictionary
        d = {'echo_id':patient, 'GT_AS':target_AS_arr, 'pred_AS':pred_AS_arr, 'logits':logits}
        df = pd.DataFrame(data=d)
        # save the dataframe
        test_results_file = os.path.join(args.finetune_save, run_name+".csv")
        df.to_csv(test_results_file)
        print("Saved results to csv.")

def test_comprehensive_tufts(args, model, loader, run_name):
    """Logs the network outputs in dataloader
    computes per-patient preds and outputs result to a DataFrame"""
    # Choose which model to evaluate.
    model.eval()
    model = model.cuda()

    with torch.set_grad_enabled(False):
        patient, img_view, target_AS_arr, pred_AS_arr, logits = [], [], [], [], []

        for data in tqdm(loader): 
            x = data['x']
            view = data['y_view']
            y = data['y_AS']
            p_id = data['p_id']

            target_AS_arr.append(y.item())
            patient.append(p_id[0])
            img_view.append(view.item())
            # Transfer data from CPU to GPU.
            x = x.cuda()
            #y = y.cuda()
            
            pred = model(x) #[n, 4] 
            prob = F.softmax(pred, dim=-1) #[n, 4]
            _, argm = torch.max(prob, dim=-1) #[n]
            # collect the model prediction info
            logits.append(prob.cpu().numpy()[0])
            pred_AS_arr.append(argm.cpu().numpy()[0])
                    
        # compile the information into a dictionary
        d = {'echo_id':patient, 'GT_AS':target_AS_arr, 'pred_AS':pred_AS_arr, 'logits':logits, 'view':img_view}
        df = pd.DataFrame(data=d)
        # save the dataframe
        test_results_file = os.path.join(args.finetune_save, run_name+".csv")
        df.to_csv(test_results_file)
        print("Saved results to csv.")
        
def finetune(args, dataloader_tr, dataloader_va, dataloader_test, run_name):

    if args.load is not None:
        print("Loading checkpoint for EchoClip pretrained on private AS_tom dataset...")
        model = ImageClassifier(embed_dim_classhead=512, dropout_prob=0.5)
        checkpoint = torch.load(Path(args.load))
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        print("Loading EchoClip...")
        model = ImageClassifier(embed_dim_classhead=512, dropout_prob=0.5)

    if args.freeze_encoder:
        print('Finetuning classification head weights only')
        for param in model.parameters():
            param.requires_grad = False
        #unfreeze classification head parameters
        for param in model.classification_head.parameters():
            param.requires_grad = True
    else:
        print("Finetuning all model weights end-to-end")
    
    print_every = 1000
    num_batches = len(dataloader_tr)
    model = model.cuda()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    #Finetuning/training
    for epoch in range(args.epochs):
        model = model.cuda()
        
        conf_AS = np.zeros((args.num_classes, args.num_classes))

        model.train()

        losses = []
        for i, batch in tqdm(enumerate(dataloader_tr)):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            cine, _, labels_AS, _ = batch
            cine = cine.squeeze(1)
            cine = cine.cuda()
            labels_AS = labels_AS.cuda()
            data_time = time.time() - start_time

            image_embeds = F.normalize(model.image_encoder(cine), dim=-1) #[b, f, embed]

            preds = model.classification_head(image_embeds) #[b, 4]
            loss = loss_fn(preds, labels_AS) #uses mean reduction 
            losses += [loss]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0: 
                percent_complete = 100 * i / len(dataloader_tr)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataloader_tr)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        loss_avg = torch.mean(torch.stack(losses))
        print(f"Loss for Train Epoch {epoch}: {loss_avg.item():.6f}")

        ## Evaluate on validation set
        print("Start validation...")
        model.eval()

        val_loss, val_AS_acc = eval(model=model, dataloader=dataloader_va, loss_fn=loss_fn, conf_AS=conf_AS)
        
        print(f"Epoch: {epoch}" f"Val Loss: {val_loss:.6f}\tVal Acc: {val_AS_acc}", flush=True)

        ## Evaluate on test set
        print("Start test...")

        test_loss, test_AS_acc = eval(model=model, dataloader=dataloader_test, loss_fn=loss_fn, conf_AS=conf_AS)

        if args.use_wandb:
            wandb.log({"tr_loss":loss_avg, 
                       "val_loss":val_loss, 
                       "val_acc":val_AS_acc,
                       "test_loss":test_loss, 
                       "test_acc":test_AS_acc,
                       "epoch":epoch
                       })
        
        print(f"Epoch: {epoch}" f"Test Loss: {test_loss:.6f}\tTest Acc: {test_AS_acc}", flush=True)

        ## Saving model
        if args.finetune_save is not None:
            dir_path = os.path.join(args.finetune_save, run_name)
            os.makedirs(dir_path, exist_ok=True)
            model_path = os.path.join(dir_path, f'checkpoint_{epoch+1}.pt')
            print('Saving model to', model_path)
            torch.save({
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path)

    # if args.finetune_save is not None:
    #     return model_path
    return model



if __name__ == '__main__':
    args = parse_arguments() 

    run_name = 'eclip_tufts_5'

    if args.dataset=='private':
        print("Getting private AS dataloader")
        dataloader_tr = get_img_dataloader(args, split='train', mode='train')
        dataloader_va = get_video_dataloader(args, split='val', mode='val')
        dataloader_test = get_video_dataloader(args, split='test', mode='val') 
    elif args.dataset=='tufts':
        print("Getting tufts dataloader")
        dataloader_eval = get_tufts_dataloader(args, split='all', mode='test')
    else:
        print(f"Dataloaders are not compatible with this dataset: {args.dataset}")

    if args.test_comprehensive is not None:
        model = ImageClassifier(embed_dim_classhead=512, dropout_prob=0.5)
        checkpoint = torch.load(Path(args.test_comprehensive))
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.dataset=='private':
            test_comprehensive_private(args=args, model=model, loader=dataloader_test, run_name=run_name)
            print(f"Completed run on {args.dataset} for checkpoint from: {args.test_comprehensive}\nrun name:{run_name}")
        else: #'tufts'
            print("Testing on tufts dataset...")
            test_comprehensive_tufts(args=args, model=model, loader=dataloader_eval, run_name=run_name)
            print(f"Completed run on {args.dataset} for checkpoint from: {args.test_comprehensive}\nrun name:{run_name}")

    else:
        if args.use_wandb:
            run = wandb.init(project="as_tab", entity="rcl_stroke", config = args, name = run_name, dir=args.wandb_dir)
    
        finetune(args, dataloader_tr, dataloader_va, dataloader_test, run_name)

        if args.use_wandb:
            wandb.finish()


