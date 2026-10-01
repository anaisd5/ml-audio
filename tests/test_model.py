import torch
from torchvision import models

from ml_audio.model import get_audio_resnet
from ml_audio.train import mixup


def test_model_initialization():
    """
    Check that the model initializes without error
    and that the final layer has the correct number of outputs.
    """
    num_classes = 10
    model = get_audio_resnet(num_classes=num_classes)

    # Check that the final layer has the correct number of outputs
    assert model.fc.out_features == num_classes


def test_model_forward_pass():
    """
    Check that a forward pass works with dummy data
    and that the output shape is correct.
    """
    num_classes = 10
    model = get_audio_resnet(num_classes=num_classes)

    # Create a dummy "scalogram"
    # Dimensions: (Batch_Size, Channels, Height, Width)
    # Here, Channels=1 (Grayscale)
    # Width=1280 (fixed width defined in dataset.py)
    dummy_input = torch.randn(2, 1, 84, 1280)

    # Forward pass
    output = model(dummy_input)

    # Checks
    assert output.shape == (
        2,
        num_classes,
    ), f"Output shape should be (2, 10), but got {output.shape}"
    assert not torch.isnan(
        output
    ).any(), "The model output contains NaNs (invalid values)"


def test_model_input_channels():
    """
    Check that the first layer accepts 1 channel (Grayscale)
    instead of 3 (standard RGB).
    """
    model = get_audio_resnet()

    # The conv1 layer should have in_channels = 1
    assert (
        model.conv1.in_channels == 1
    ), "The first layer should expect 1 channel (audio), not 3."


def test_model_input_layer_pretrained():
    """
    Check that the first layer keeps the pretrained filters
    (summed over the 3 RGB channels) instead of random ones.
    """
    model = get_audio_resnet()
    pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    expected = pretrained.conv1.weight.data.sum(dim=1, keepdim=True)
    assert torch.allclose(
        model.conv1.weight.data, expected
    ), "The first layer should be initialised with the pretrained weights."


def test_mixup():
    """
    Check that mixup mixes each example with another one of the batch
    with a proportion between 0 and 1.
    """
    inputs = torch.randn(8, 1, 84, 129)
    labels = torch.arange(8)

    mixed, labels_a, labels_b, lam = mixup(inputs, labels)

    assert mixed.shape == inputs.shape
    assert 0 <= lam <= 1
    assert torch.equal(labels_a, labels)
    # labels_b gives the order of the other examples
    expected = lam * inputs + (1 - lam) * inputs[labels_b]
    assert torch.allclose(mixed, expected)


def test_model_frozen_layers():
    """
    Check that the frozen layers are not trainable and that the other
    layers still are.
    """
    model = get_audio_resnet(frozen_layers=2)

    frozen = [model.conv1, model.bn1, model.layer1, model.layer2]
    trainable = [model.layer3, model.layer4, model.fc]

    for module in frozen:
        assert not any(param.requires_grad for param in module.parameters())
    for module in trainable:
        assert all(param.requires_grad for param in module.parameters())
