# model.py
from torch import nn
from torchvision.models import resnet, efficientnet_b0
import time
from config import config

class AgeEstimationModel(nn.Module):
    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(AgeEstimationModel, self).__init__()
        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=2048, out_features=256, bias=True),
                nn.Linear(in_features=256, out_features=self.output_nodes, bias=True),
            )
        elif model_name == 'efficientnet':
            self.model = efficientnet_b0(weights=None)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=256, bias=True),
                nn.Linear(in_features=256, out_features=self.output_nodes, bias=True),
            )
        elif model_name == 'vit':
            img_size = int(config.get('img_size', config.get('imgwidth', 128)))
            self.model = timm.create_model(
                'vit_small_patch14_dinov2.lvd142m',
                img_size=img_size,
                pretrained=bool(pretrain_weights),
            )
            num_features = 384
            self.model.head = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Linear(256, self.output_nodes),
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        return self.model(x)

def num_trainable_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
