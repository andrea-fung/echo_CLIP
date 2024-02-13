### Adapted from mlfoundations/wise-ft repo

import os
import copy
import time
from tqdm import tqdm
import numpy as np
import wandb
import torch

import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score

from args import parse_arguments
from open_clip import tokenize
from models.modeling import ImageClassifier
from models.utils import cosine_lr, torch_load, LabelSmoothing
from utils import update_confusion_matrix, acc_from_confusion_matrix
from dataloader import get_img_dataloader, get_video_dataloader
from losses import ClipLoss

def eval(model, dataloader, loss_fn, conf_AS):
    losses = []
    preds = []
    labels = []
    ## count = 0
    with torch.set_grad_enabled(False):
        for cine, _, label_AS in tqdm(dataloader):
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

def finetune(args, dataloader_tr, dataloader_va, dataloader_test):
    #assert args.load is not None, "Please provide the path to a checkpoint through --load."
    #assert args.train_dataset is not None, "Please provide a training dataset."

    model = ImageClassifier(embed_dim_classhead=512)

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

            cine, tab_data, labels_AS = batch
            cine = cine.squeeze(1)
            cine = cine.cuda()
            ## tab_data = tab_data.cuda()
            labels_AS = labels_AS.cuda()
            data_time = time.time() - start_time

            image_embeds = F.normalize(model.image_encoder(cine), dim=-1) #[b, f, embed]

            ## We'll use the CLIP BPE tokenizer to tokenize the prompts
            tab_prompts = tokenize(tab_data).cuda()
            ## Now we can encode the prompts into embeddings
            tab_on = False
            if tab_on:
                tab_embeds = F.normalize(model.text_encoder(tab_prompts), dim=-1)

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

        loss_avg = torch.mean(torch.stack(losses)).item()

        ## Evaluate on validation set
        print("Start validation...")
        model.eval()

        val_loss, val_AS_acc = eval(model=model, dataloader=dataloader_va, loss_fn=loss_fn, conf_AS=conf_AS)

        if args.use_wandb:
            wandb.log({"tr_loss":loss_avg, "val_loss":val_loss, "val_acc":val_AS_acc})
        
        print(f"Epoch: {epoch}" f"Val Loss: {val_loss:.6f}\tVal Acc: {val_AS_acc}", flush=True)

        ## Evaluate on test set
        print("Start test...")

        test_loss, test_AS_acc = eval(model=model, dataloader=dataloader_test, loss_fn=loss_fn, conf_AS=conf_AS)

        if args.use_wandb:
            wandb.log({"test_loss":test_loss, "test_acc":test_AS_acc})
        
        print(f"Epoch: {epoch}" f"Test Loss: {test_loss:.6f}\tTest Acc: {test_AS_acc}", flush=True)

        ## Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            print('Saving model to', model_path)
            model.save(model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments() 

    run_name = "echoclip_finetune_all"
    if args.use_wandb:
        run = wandb.init(project="as_tab", entity="rcl_stroke", config = args, name = run_name)

    dataloader_tr = get_img_dataloader(args, split='train', mode='train')
    dataloader_va = get_video_dataloader(args, split='val', mode='val')
    dataloader_test = get_video_dataloader(args, split='test', mode='val') 
     
    finetune(args, dataloader_tr, dataloader_va, dataloader_test)

    if args.use_wandb:
        wandb.finish()


#CLIP loss 
# clip_loss = ClipLoss(local_loss=args.local_loss,
#                     gather_with_grad=args.gather_with_grad,
#                     cache_labels=True,
#                     rank=args.rank,
#                     world_size=args.world_size,
#                     use_horovod=args.horovod)
# temperature = 1.0
# logits = (tab_embeds @ video_embeds.T) / temperature
# images_similarity = video_embeds @ video_embeds.T
# texts_similarity = tab_embeds @ tab_embeds.T
# targets = F.softmax(
#     (images_similarity + texts_similarity) / 2 * temperature, dim=-1)
# texts_loss = F.cross_entropy(logits, targets, reduction='none')
# images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
# loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
# loss = loss.mean()