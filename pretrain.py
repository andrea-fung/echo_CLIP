### Adapted from mlfoundations/wise-ft repo

import os
import time
from tqdm import tqdm
import numpy as np
import wandb
import torch

import torch.nn as nn
import torch.nn.functional as F

from get_config import get_config
from open_clip import tokenize
from models.modeling import ImageClassifier
from models.utils import cosine_lr
from dataloader import get_video_dataloader
from losses import ClipLoss

from open_clip import create_model_and_transforms

def pretrain(configs, dataloader_tr, dataloader_val, run_name, init_logit_scale=np.log(1 / 0.07)):
    #assert configs['load'] is not None, "Please provide the path to a checkpoint through --load."
    #assert configs['train_dataset'] is not None, "Please provide a training dataset."

    echoclip = ImageClassifier(embed_dim_classhead=512, dropout_prob=0.5)

    for param in echoclip.parameters():
        param.requires_grad = True
    
    print_every = 1000
    num_batches = len(dataloader_tr)
    echoclip = echoclip.cuda()

    params = [p for p in echoclip.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config["wd"])

    scheduler = cosine_lr(optimizer, config['lr'], config["warmup_length"], config["epochs"] * num_batches)

    for epoch in range(config["pretrain_epochs"]):
        echoclip = echoclip.cuda()
        
        echoclip.train()

        losses = []
        for i, batch in tqdm(enumerate(dataloader_tr)):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            cine, report_data, _, _ = batch
            cine = cine.squeeze(1)
            cine = cine.cuda()
            data_time = time.time() - start_time

            b, f, c, h, w = cine.shape
            #Remove batch dimension
            cine = cine.reshape(-1, c, h, w) #[f, c, h, w]
            video_embed = echoclip.model.encode_image(cine)

            #temporal pooling across frames
            video_embed = video_embed.mean(dim=0, keepdim=True) #[1, e] ##TODO - FIXBUG
            #video_embed = F.normalize(echoclip.model.encode_image(cine), dim=-1) #[1, embed]
            print(f"output shape of normalized video embedding: {video_embed.shape}")
            ## We'll use the CLIP BPE tokenizer to tokenize the prompts
            text_prompts = tokenize(report_data).cuda()
            ## Now we can encode the prompts into embeddings
            text_embed = F.normalize(echoclip.model.encode_text(text_prompts), dim=-1)

            #CLIP loss 
            clip_loss = ClipLoss() #mean reduction
            logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
            loss = clip_loss(image_features=video_embed, text_features=text_embed, logit_scale=logit_scale)
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


    #     print("Starting validation...")
    #     echoclip.eval()

    #     with torch.set_grad_enabled(False):
    #         val_losses = []
    #         for cine, report_data, labels_AS, _ in tqdm(dataloader_val):
    #             ## count+=1
    #             cine = cine.squeeze(1)
    #             cine = cine.cuda()
    #             labels_AS = labels_AS.cuda()
            
    #             image_embeds = F.normalize(echoclip.model.image_encoder(cine), dim=-1) #[b, f, embed]

    #             ## We'll use the CLIP BPE tokenizer to tokenize the prompts
    #             text_prompts = tokenize(report_data).cuda()
    #             ## Now we can encode the prompts into embeddings
    #             tab_embeds = F.normalize(echoclip.model.text_encoder(text_prompts), dim=-1)

    #             #CLIP loss 
    #             clip_loss = ClipLoss() #mean reduction
    #             logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
    #             loss = clip_loss(image_features=image_embeds, text_features=tab_embeds, logit_scale=logit_scale)
    #             val_losses += [loss]
        
    #     val_loss = torch.mean(torch.stack(val_losses))
    #     print(f"Epoch: {epoch}" f"Val Loss: {val_loss}", flush=True)
    #     if config["use_wandb"]:
    #         wandb.log({"tr_loss":loss_avg, "val_loss":val_loss})
            
    #     ## Saving model
    #     if config['pretrain_save'] is not None:
    #         dir_path = os.path.join(config['pretrain_save'], run_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         model_path = os.path.join(dir_path, f'checkpoint_{epoch+1}.pt')
    #         print('Saving model to', model_path)
    #         torch.save({'model': echoclip.model.state_dict()}, model_path)

    # if config['pretrain_save'] is not None:
    #     return model_path


if __name__ == '__main__':
    config = get_config()

    run_name = config["exp_name"]
    config["type_of_training"] = "pretraining"

    if config["use_wandb"]:
        run = wandb.init(project="as_tab", entity="rcl_stroke", config = config, name = run_name, dir=config["wandb_dir"])

    dataloader_tr = get_video_dataloader(config, split='train', mode='train')
    dataloader_val = get_video_dataloader(config, split='val', mode='val')
    
    pretrain(config, dataloader_tr, dataloader_val, run_name)

    # if config["use_wandb"]:
    #     wandb.finish()