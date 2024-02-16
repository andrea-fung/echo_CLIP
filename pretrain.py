### Adapted from mlfoundations/wise-ft repo

import os
import time
from tqdm import tqdm
import numpy as np
import wandb
import torch

import torch.nn as nn
import torch.nn.functional as F

from args import parse_arguments
from open_clip import tokenize
from models.modeling import ImageClassifier
from models.utils import cosine_lr
from dataloader import get_img_dataloader
from losses import ClipLoss

def pretrain(args, dataloader_tr, dataloader_val, run_name, init_logit_scale=np.log(1 / 0.07)):
    #assert args.load is not None, "Please provide the path to a checkpoint through --load."
    #assert args.train_dataset is not None, "Please provide a training dataset."

    model = ImageClassifier(embed_dim_classhead=512, dropout_prob=0.5)
    
    print_every = 1000
    num_batches = len(dataloader_tr)
    model = model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    for epoch in range(args.pretrain_epochs):
        model = model.cuda()
        
        model.train()

        losses = []
        for i, batch in tqdm(enumerate(dataloader_tr)):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            cine, tab_data, labels_AS = batch
            cine = cine.squeeze(1)
            cine = cine.cuda()
            tab_data = tab_data #str input
            labels_AS = labels_AS.cuda()
            data_time = time.time() - start_time

            image_embeds = F.normalize(model.image_encoder(cine), dim=-1) #[b, f, embed]

            ## We'll use the CLIP BPE tokenizer to tokenize the prompts
            tab_prompts = tokenize(tab_data).cuda()
            ## Now we can encode the prompts into embeddings
            tab_embeds = F.normalize(model.text_encoder(tab_prompts), dim=-1)

            #CLIP loss 
            clip_loss = ClipLoss() #mean reduction
            logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
            loss = clip_loss(image_features=image_embeds, text_features=tab_embeds, logit_scale=logit_scale)
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
        print(f"Avg. Loss for Train Epoch {epoch}: {loss_avg.item():.6f}")


        print("Starting validation...")
        model.eval()

        with torch.set_grad_enabled(False):
            val_losses = []
            for cine, tab_data, labels_AS in tqdm(dataloader_val):
                ## count+=1
                cine = cine.squeeze(1)
                cine = cine.cuda()
                tab_data = tab_data
                labels_AS = labels_AS.cuda()
            
                image_embeds = F.normalize(model.image_encoder(cine), dim=-1) #[b, f, embed]

                ## We'll use the CLIP BPE tokenizer to tokenize the prompts
                tab_prompts = tokenize(tab_data).cuda()
                ## Now we can encode the prompts into embeddings
                tab_embeds = F.normalize(model.text_encoder(tab_prompts), dim=-1)

                #CLIP loss 
                clip_loss = ClipLoss() #mean reduction
                logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
                loss = clip_loss(image_features=image_embeds, text_features=tab_embeds, logit_scale=logit_scale)
                val_losses += [loss]
        
        val_loss = torch.mean(torch.stack(val_losses))
        print(f"Epoch: {epoch}" f"Val Loss: {val_loss}", flush=True)
        if args.use_wandb:
            wandb.log({"tr_loss":loss_avg, "val_loss":val_loss})
            
        ## Saving model
        if args.pretrain_save is not None:
            dir_path = os.path.join(args.pretrain_save, run_name)
            os.makedirs(dir_path, exist_ok=True)
            model_path = os.path.join(dir_path, f'checkpoint_{epoch+1}.pt')
            print('Saving model to', model_path)
            torch.save({'model': model.state_dict()}, model_path)

    if args.pretrain_save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments() 

    run_name = "eclip_pretrain"
    if args.use_wandb:
        run = wandb.init(project="as_tab", entity="rcl_stroke", config = args, name = run_name, dir=args.wandb_dir)

    dataloader_tr = get_img_dataloader(args, split='train', mode='train')
    dataloader_val = get_img_dataloader(args, split='val', mode='val')
     
    pretrain(args, dataloader_tr, dataloader_val, run_name)

    if args.use_wandb:
        wandb.finish()