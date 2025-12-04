import torch
from ml_audio.model import get_audio_resnet


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
