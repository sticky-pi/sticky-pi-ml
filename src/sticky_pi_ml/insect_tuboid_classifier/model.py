import torch
import torch.nn as nn
import torch.utils.data
from typing import Dict
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

# the resnet architectures,
# fixme, only resnet-50 is supported for now

RESNET_VARIANTS = {
    '50': {'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
           'block': Bottleneck,
           'layers': [3, 4, 6, 3]},

    '152': {'url': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
           'block': Bottleneck,
           'layers': [3, 8, 36, 3]}
}


class ResNetPlus(ResNet):
    n_extra_dimension = 1

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__(block, layers, num_classes, zero_init_residual,
                         groups, width_per_group,
                         replace_stride_with_dilation,
                         norm_layer)
        # two modifications to the FC layer:
        # 1. the input has one extra dimension describing the size of the animal
        # 2. the output corresponds to the number of labels/classes
        self.fc = nn.Linear(self.n_extra_dimension + 512 * block.expansion, num_classes)

    def _make_features(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Makes a batch of enhance features (custom features + resnet features)
        Implementation derived from torchvision.models.resnet.ResNet._forward_impl().

        :param x: N input images as a batch of 3d tensors shape = ([N x] 224 x 224 x 3)
        :param s: the scaling factor of the image compared to the original, as a batch of float tensors
        :return: the vector of features to be fed to the FC layer ([N x] (1 + 512 * block.expansion))
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # the only difference vs the ref implementation is here, we just add the scale as an extra element in the output
        # feature vectors
        x = torch.cat((x, s), 1)
        return x

    def _forward_impl(self, inputs: Dict) -> torch.Tensor:
        """
        Batches have size N. Instead of running reset on single images, we compute resnet features over M shots of the
        same instance. Then, we compute an average (median) feature vector per instance.

        :param inputs: a dictionary of inputs, already arranged in a batch of size N. Keys:
            * ``'array'`` tensor of shape ([N x] M x 224 x 224, 3)
            * ``'scale'`` tensor of shape ([N x] M x 1)
        :return: a tensor of labels ([N x] 1)
        """
        batch_size, n_shots, h, w, depth = inputs['array'].shape

        instance_features = []
        # over all shots through the batch dimension
        for i in range(n_shots):
            x = inputs['array'][:, i, :, :, :].reshape((batch_size, h, w, depth))
            s = inputs['scale'][:, i, :].reshape((batch_size, 1))
            x = self._make_features(x, s)
            instance_features.append(x)
        # then we stack the features and compute their medians (per vector element and instance)
        avg_features = torch.median(torch.stack(instance_features, dim=1), dim=1)[0]
        # avg_features is therefore of size ([N x] (self._n_extra_dimension + 512 * block.expansion))
        o = self.fc(avg_features)
        return o

    def forward(self, x):
        return self._forward_impl(x)


def make_resnet(pretrained: bool, n_classes, progress=True, resnet_variant: str = '50'):
    try:
        variant = RESNET_VARIANTS[resnet_variant]
    except KeyError:
        raise KeyError(
            'Unsupported variant of resnet %s. Available options are %s' % (resnet_variant, RESNET_VARIANTS.keys()))

    model = ResNetPlus(variant['block'], variant['layers'])

    # the actual number of feature to use in the FC layer in OUR modified resnet
    n_features = model.fc.in_features

    if pretrained:
        from torch.hub import load_state_dict_from_url
        # load the pretrained weights of the stock resnet
        state_dict = load_state_dict_from_url(variant['url'], progress=progress)
        # these will not have the same dimensions as OUR resnet (in which we enhance the feature vector)
        # therefore, we modify OUR model to match the stock resnet before we can load the weights
        model.fc = nn.Linear(n_features - ResNetPlus.n_extra_dimension, 1000)
        model.load_state_dict(state_dict)
    # After loading the weights, we ensure that the last layer (FC) outputs match the number of classes
    model.fc = nn.Linear(n_features, n_classes)

    return model

