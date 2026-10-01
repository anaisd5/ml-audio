import torch.nn as nn
from torchvision import models


def get_audio_resnet(num_classes=10, frozen_layers=0):
    """
    Create a ResNet-18 model adapted to audio scalograms. The scalograms
    expected are first processed with GTZANDataset methods from dataset.py.
    It expects only one channel (the first dimension of the tensor).

    :param num_classes: the number of output classes
                        (e.g. 10 for genres of GTZAN dataset)
    :type num_classes: int
    :param frozen_layers: the number of groups of layers (from 0 to 4)
                          that are frozen, i.e. not modified by the
                          training. The ResNet-18 has 4 groups (layer1 to
                          layer4): with 2, the first layer (conv1), layer1
                          and layer2 keep their pretrained weights and
                          only layer3, layer4 and the output are trained.
    :type frozen_layers: int
    :raises ValueError: if frozen_layers is not between 0 and 4
    :returns: modified ResNet model
    :rtype: torch.nn.Module
    """

    # Load a pre-trained ResNet-18 model

    # load the weights (transfer learning)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the first layer (conv1)
    # The layer 'conv1' originally expects 3 channels (in_channels=3),
    # it is replaced by a layer that accepts 1 channel
    pretrained_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,  # number of filters (and then output channels)
        kernel_size=(7, 7),  # size of the filter (H x L)
        stride=(2, 2),  # the number of pixels the filter moves over
        padding=(3, 3),  # the number of pixels to add around the image
        bias=False,  # no bias (an additional parameter)
    )
    # Keep the pretrained filters instead of random ones: the weights
    # of the 3 channels (RGB) are summed into 1 channel
    model.conv1.weight.data = pretrained_conv1.weight.data.sum(
        dim=1, keepdim=True
    )

    # Modify the last layer (output)
    # The fully connected orignal layer outputed 1000 classes (ImageNet),
    # it is replaced by a layer that outputs 'num_classes_ (here 10)

    # store the number of features received by this layer
    num_ftrs = model.fc.in_features
    # define a new output layer receiving as many features as the old layer
    # and outputs 'num_classes'
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Freeze the first layers
    # With few training data, the first layers (which detect simple
    # shapes) are kept as they are to limit overfitting
    if not 0 <= frozen_layers <= 4:
        raise ValueError("frozen_layers should be between 0 and 4")

    if frozen_layers > 0:
        groups = [model.layer1, model.layer2, model.layer3, model.layer4]
        frozen_modules = [model.conv1, model.bn1] + groups[:frozen_layers]
        for module in frozen_modules:
            for param in module.parameters():
                # the optimiser will not modify this parameter
                param.requires_grad = False

    return model
