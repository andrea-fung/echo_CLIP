### Adapted from mlfoundations/wise-ft repo

import torch
import torch.nn as nn
import copy

#import clip.clip as clip
from open_clip import create_model_and_transforms

import models.utils as utils


# class ImageEncoder(torch.nn.Module):
#     def __init__(self, keep_lang=False):
#         super().__init__()

#         # self.model, self.train_preprocess, self.val_preprocess = clip.load(
#         #     args.model, args.device, jit=False)
#         self.model, self.train_preprocess, self.val_preprocess = create_model_and_transforms(
#             "hf-hub:mkaichristensen/echo-clip", precision="float32", device="cuda")

#         self.text_encoder = self.model.encode_text
        
#         #self.cache_dir = args.cache_dir

#         if not keep_lang and hasattr(self.model, 'transformer'):
#             delattr(self.model, 'transformer')

#     def forward(self, images):
#         assert self.model is not None
#         return self.model.encode_image(images)

#     def save(self, filename):
#         print(f'Saving image encoder to {filename}')
#         utils.torch_save(self, filename)

#     @classmethod
#     def load(cls, filename):
#         print(f'Loading image encoder from {filename}')
#         return utils.torch_load(filename)



class ImageClassifier(torch.nn.Module):
    def __init__(self, embed_dim_classhead, dropout_prob, process_images=True):
        super().__init__()
        self.model, self.train_preprocess, self.val_preprocess = create_model_and_transforms(
            "hf-hub:mkaichristensen/echo-clip", precision="float32", device="cuda")
        self.image_encoder = self.model.encode_image
        self.text_encoder = self.model.encode_text
        self.process_images = process_images
        # if self.image_encoder is not None:
        #     self.train_preprocess = self.image_encoder.train_preprocess
        #     self.val_preprocess = self.image_encoder.val_preprocess
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=embed_dim_classhead, out_features=embed_dim_classhead//2, bias=True),
            nn.LayerNorm(embed_dim_classhead//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=embed_dim_classhead//2, out_features=4, bias=True))

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)

        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

# if __name__ == '__main__':
#     #run image encoder and save to filename
#     encoder = ImageEncoder()
#     encoder.save('/workspace/echo_CLIP/logs/image_encoder.pt')