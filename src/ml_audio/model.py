import torch.nn as nn
from torchvision import models

def get_audio_resnet(num_classes=10):
    """
    Create a ResNet-18 model adapted to audio scalograms. The scalograms
    expected are first processed with GTZANDataset methods from dataset.py. 
    It expects only one channel (the first dimension of the tensor).

    Args:
        num_classes (int): The number of output classes
                           (e.g. 10 for genres of GTZAN dataset).
    
    Returns:
        torch.nn.Module: The modified ResNet model.
    """
    
    # Load a pre-trained ResNet-18 model 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # load the weights (transfer learning)

    # Modify the first layer (conv1)
    # The layer 'conv1' originally expects 3 channels (in_channels=3), 
    # it is replaced by a layer that accepts 1 channel
    model.conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=64,        # number of filters (and then output channels)
        kernel_size=(7, 7),     # size of the filter (H x L)
        stride=(2, 2),          # the number of pixels the filter moves over
        padding=(3, 3),         # the number of pixels to add around the 'image'
        bias=False              # no bias (an additional parameter)
    )

    # Modify the last layer (output)
    # The fully connected orignal layer outputed 1000 classes (ImageNet), 
    # it is replaced by a layer that outputs 'num_classes_ (here 10)
    num_ftrs = model.fc.in_features             # store the number of features received by this layer
    model.fc = nn.Linear(num_ftrs, num_classes) # define a new output layer receiving as many features as the old layer
                                                # and outputs 'num_classes'

    return model