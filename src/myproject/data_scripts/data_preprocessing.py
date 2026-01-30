import torch
from myproject.data_scripts.load_data import load_MNIST, load_CIFAR10

def process_MNIST(ConvolutionalLayer=False):
    """
    Load and preprocess the MNIST dataset for training.

    This function loads the MNIST dataset using `load_MNIST()`, normalizes the
    image pixel values to the range [0, 1], and returns the images and labels
    for both training and testing.

    The `ConvolutionalLayer` flag controls whether images are returned as 2D
    tensors (for CNNs) or flattened vectors (for fully connected networks).

    Args:
        ConvolutionalLayer (bool, optional): If True, the images are kept in
            their original 2D shape (N, 28, 28) suitable for convolutional
            layers. If False, images are flattened to (N, 784). Defaults to False.

    Returns:
        tuple: (train_images, train_targets, test_images, test_targets)
            train_images (torch.Tensor or np.ndarray): Normalized training images.
            train_targets (list or torch.Tensor): Training labels.
            test_images (torch.Tensor or np.ndarray): Normalized test images.
            test_targets (list or torch.Tensor): Test labels.

    Notes:
        - The returned images are normalized by dividing by 255.0.
        - For MNIST, the dataset stores images as tensors internally.
        - If `ConvolutionalLayer` is False, images are reshaped to vectors for
          fully connected models.
    """

    train_data, test_data = load_MNIST()

    train_images = train_data.data / 255.0
    test_images = test_data.data / 255.0

    train_targets = train_data.targets 
    test_targets = test_data.targets 

    if not ConvolutionalLayer:
        train_images = train_images.reshape(-1, 28*28)
        test_images = test_images.reshape(-1, 28*28) 

    return train_images, train_targets, test_images, test_targets


def process_CIFAR10(ConvolutionalLayer=True):
    """
    This project does not yet efficiently handle the CIFAR-10 dataset.
    Nonetheless, it is still possible to load the dataset using this
    function just like we did with the MNIST dataset for experimental purposes.
    Future versions of this framework aim to solve this. 

    Load and preprocess CIFAR-10 dataset.

    Images are normalized to [0, 1] and converted to torch tensors.
    The image shape is converted to (N, C, H, W) for CNNs.

    Args:
        convolutional_layer (bool): If True, keep image shape as (N, C, H, W).
                                    If False, flatten images to (N, 3072).

    Returns:
        tuple: (train_images, train_targets, test_images, test_targets)
    """

    train_data, test_data = load_CIFAR10()

    train_images = torch.tensor(train_data.data, dtype=torch.float32) / 255.0
    train_images = train_images.permute(0, 3, 1, 2)

    test_images = torch.tensor(test_data.data, dtype=torch.float32) / 255.0
    test_images = test_images.permute(0, 3, 1, 2)

    train_targets = torch.tensor(train_data.targets, dtype=torch.long)
    test_targets = torch.tensor(test_data.targets, dtype=torch.long)

    if not ConvolutionalLayer:
        train_images = train_images.reshape(train_images.size(0), -1)
        test_images = test_images.reshape(test_images.size(0), -1)

    return train_images, train_targets, test_images, test_targets
