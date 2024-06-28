# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
from .backbones import get_backbone
from copy import deepcopy
import torch.nn.functional as F

BLOCKNAMES = {
    "resnet": {
        "stem": ["conv1", "bn1", "relu", "maxpool"],
        "block1": ["layer1"],
        "block2": ["layer2"],
        "block3": ["layer3"],
        "block4": ["layer4"],
    },
    "clipresnet": {
        "stem": ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "relu", "avgpool"],
        "block1": ["layer1"],
        "block2": ["layer2"],
        "block3": ["layer3"],
        "block4": ["layer4"],
    },
    "clipvit": {  # vit-base
        "stem": ["conv1"],
        "block1": ["transformer.resblocks.0", "transformer.resblocks.1", "transformer.resblocks.2"],
        "block2": ["transformer.resblocks.3", "transformer.resblocks.4", "transformer.resblocks.5"],
        "block3": ["transformer.resblocks.6", "transformer.resblocks.7", "transformer.resblocks.8"],
        "block4": ["transformer.resblocks.9", "transformer.resblocks.10", "transformer.resblocks.11"],
    },
    "regnety": {
        "stem": ["stem"],
        "block1": ["trunk_output.block1"],
        "block2": ["trunk_output.block2"],
        "block3": ["trunk_output.block3"],
        "block4": ["trunk_output.block4"]
    },
    "convnext": {
        "stem":["stem"], 
        "stage0": ["block0", "block1", "block2"],
        "stage1": ["block0", "block1", "block2"],
        "stage2": ["block0", "block1", "block2", "block3", "block4", "block5", "block6", "block7", "block8"],
        "stage3": ["block0", "block1", "block2"],
    }
}


def get_module(module, name):
    for n, m in module.named_modules():
        if n == name:
            return m


def build_blocks(model, block_name_dict):
    #  blocks = nn.ModuleList()
    blocks = []  # saved model can be broken...
    for _key, name_list in block_name_dict.items():
        block = nn.ModuleList()
        for module_name in name_list:
            module = get_module(model, module_name)
            block.append(module)
        blocks.append(block)

    return blocks


def freeze_(model):
    """Freeze model
    Note that this function does not control BN
    """
    for p in model.parameters():
        p.requires_grad_(False)


class URResNet(torch.nn.Module):
    """ResNet + FrozenBN + IntermediateFeatures
    """

    def __init__(self, input_shape, hparams, preserve_readout=False, freeze=None, feat_layers=None):
        assert input_shape == (3, 224, 224), input_shape
        super().__init__()

        self.network, self.n_outputs = get_backbone(hparams.model, preserve_readout, hparams.pretrained)

        if hparams.model == "resnet18":
            block_names = BLOCKNAMES["resnet"]
        elif hparams.model.startswith("resnet50"):
            block_names = BLOCKNAMES["resnet"]
        elif hparams.model.startswith("clip_resnet"):
            block_names = BLOCKNAMES["clipresnet"]
        elif hparams.model.startswith("clip_vit"):
            block_names = BLOCKNAMES["clipvit"]
        elif hparams.model == "swag_regnety_16gf":
            block_names = BLOCKNAMES["regnety"]
        elif hparams.model.startswith("vit"):
            block_names = BLOCKNAMES["vit"]
        else:
            raise ValueError(hparams.model)

        self._features = []
        self.feat_layers = self.build_feature_hooks(feat_layers, block_names)
        self.blocks = build_blocks(self.network, block_names)

        self.freeze(freeze)

        if not preserve_readout:
            self.dropout = nn.Dropout(hparams["resnet_dropout"])
        else:
            self.dropout = nn.Identity()
            assert hparams["resnet_dropout"] == 0.0

        self.hparams = hparams
        self.freeze_bn()

    def freeze(self, freeze):
        if freeze is not None:
            if freeze == "all":
                freeze_(self.network)
            else:
                for block in self.blocks[:freeze+1]:
                    freeze_(block)

    def hook(self, module, input, output):
        self._features.append(output)

    def build_feature_hooks(self, feats, block_names):
        assert feats in ["stem_block", "block"]

        if feats is None:
            return []

        # build feat layers
        if feats.startswith("stem"):
            last_stem_name = block_names["stem"][-1]
            feat_layers = [last_stem_name]
        else:
            feat_layers = []

        for name, module_names in block_names.items():
            if name == "stem":
                continue

            module_name = module_names[-1]
            feat_layers.append(module_name)

        #  print(f"feat layers = {feat_layers}")

        for n, m in self.network.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)

        return feat_layers

    def forward(self, x, ret_feats=False):
        """Encode x into a feature vector of size n_outputs."""
        self.clear_features()
        out = self.dropout(self.network(x))
        if ret_feats:
            return out, self._features
        else:
            return out

    def clear_features(self):
        self._features.clear()

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class URConvNeXt(torch.nn.Module):
    """ConvNeXt + FrozenBN + IntermediateFeatures
    """

    def __init__(self, input_shape, hparams, preserve_readout=False, freeze=None, feat_layers=None):
        assert input_shape == (5, 224, 224), input_shape
        super().__init__()

        self.network, self.n_outputs = get_backbone(hparams.model, preserve_readout, hparams.pretrained)
        first_layer = self._get_first_layer(self.network, 5)
        self.network =  nn.Sequential(
            first_layer,
            self.network.stem[1],
            self.network.stages[0],
            self.network.stages[1],
            self.network.stages[2].downsample,
            *[self.network.stages[2].blocks[i] for i in range(9)],
            self.network.stages[3].downsample,
            *[self.network.stages[3].blocks[i] for i in range(3)],
        )
    
        self._features = []
        block_names = BLOCKNAMES["convnext"]
        #self.feat_layers = self.build_feature_hooks(feat_layers, block_names)
        self.blocks = build_blocks(self.network, block_names)
        self.feat_layers = [
            "0.blocks.2.drop_path",
            "1.blocks.2.drop_path",
            "11.drop_path",
            "15.drop_path",
        ]
        self.build_feature_hooks(self.feat_layers)

        self.freeze(freeze)

        self.hparams = hparams
        self.freeze_bn()

    def freeze(self, freeze):
        if freeze is not None:
            if freeze == "all":
                freeze_(self.network)
            else:
                for block in self.blocks[:freeze+1]:
                    freeze_(block)

    def hook(self, module, input, output):
        self._features.append(output)

    def build_feature_hooks(self, feat_layers):
        for n, m in self.network.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)
        return None

    def _get_first_layer(self, model, new_in_dim):
        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        new_shape = (out_dim, new_in_dim, kh, kw)
        layer_1 = nn.Parameter(torch.zeros(new_shape))
        nn.init.kaiming_normal_(layer_1, mode="fan_out", nonlinearity="relu")
        conv1 = deepcopy(model.stem[0])
        conv1.weight = layer_1
        return conv1
    
    def forward(self, x, ret_feats=False):
        """Encode x into a feature vector of size n_outputs."""
        self.clear_features()
        out = self.network(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if ret_feats:
            return out, self._features
        else:
            return out

    def clear_features(self):
        self._features.clear()

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def URFeaturizer(input_shape, hparams, **kwargs):
    """Auto-select an appropriate featurizer for the given input shape."""
    if input_shape[0] == 5:
        return URConvNeXt(input_shape, hparams, **kwargs)
    elif input_shape[1:3] == (224, 224):
        return URResNet(input_shape, hparams, **kwargs)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")
