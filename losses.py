import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipLoss(nn.Module):

    def __init__(
            self,
            cache_labels=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits:
            labels = torch.arange(num_logits, dtype=torch.long).cuda()
            if self.cache_labels:
                self.labels = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        #device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(logits_per_image.shape[0]) #[0...N-1]

        #Maximizing similarity along the diagonal of cosine sim matrix by aligning logits_per_image/logits_per_text
        #with diagonal entries.
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss