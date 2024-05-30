### Adapted from mlfoundations/wise-ft repo

import torch
import torch.nn as nn
import copy

#import clip.clip as clip
from open_clip import create_model_and_transforms

import models.utils as utils

class ImageClassifier(torch.nn.Module):
    def __init__(self, embed_dim_classhead, dropout_prob, process_images=True):
        super().__init__()
        self.model, self.train_preprocess, self.val_preprocess = create_model_and_transforms(
            "hf-hub:mkaichristensen/echo-clip", precision="float32", device="cuda")
        # self.image_encoder = self.model.encode_image
        # self.text_encoder = self.model.encode_text
        self.process_images = process_images
        
        # self.classification_head = nn.Sequential(
        #     nn.Linear(in_features=embed_dim_classhead, out_features=embed_dim_classhead//2, bias=True),
        #     nn.LayerNorm(embed_dim_classhead//2),
        #     nn.LeakyReLU(negative_slope=0.05, inplace=True),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(in_features=embed_dim_classhead//2, out_features=4, bias=True))

    def forward(self, inputs): #[f, c, h, w]
        if self.process_images:
            f, c, h, w = inputs.shape
            frame_outputs = self.model.encode_image(inputs) #[f, e]
            print(f"shape of image encoder output: {outputs.shape}")
            #temporal pooling across frames
            video_outputs = torch.mean(outputs, dim=0, keepdim=True)
        #outputs = self.classification_head(inputs)

        return video_outputs

    # def save(self, filename):
    #     print(f'Saving image classifier to {filename}')
    #     utils.torch_save(self, filename)

    # @classmethod
    # def load(cls, filename):
    #     print(f'Loading image classifier from {filename}')
    #     return utils.torch_load(filename)

# if __name__ == '__main__':
#     #run image encoder and save to filename
#     encoder = ImageEncoder()
#     encoder.save('/workspace/echo_CLIP/logs/image_encoder.pt')