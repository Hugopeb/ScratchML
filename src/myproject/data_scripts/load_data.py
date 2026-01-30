from torchvision import datasets 
from torchvision.transforms import ToTensor

from myproject.config import DATA_DIR

def load_MNIST():
    """
    Download and load the MNIST dataset using torchvision.

    This function returns two dataset objects (train and test), with images
    converted to torch tensors using the `ToTensor()` transform.

    Notes:
    - The datasets are downloaded to `DATA_DIR / "train_MNIST"` and
      `DATA_DIR / "test_MNIST"`.
    - The returned objects are instances of `torchvision.datasets.MNIST`.
    - Data is not preloaded into memory. Images are converted to tensors only
      when accessed via indexing (e.g., dataset[i]).

    Returns:
        tuple: (train_dataset, test_dataset)
            train_dataset (torchvision.datasets.MNIST): Training set.
            test_dataset (torchvision.datasets.MNIST): Test set.
    """

    train_MNIST = datasets.MNIST(
        root = DATA_DIR / "train_MNIST",
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_MNIST = datasets.MNIST(
        root = DATA_DIR / "test_MNIST",
        train = False,
        transform = ToTensor(),
        download = True
    )

    return train_MNIST, test_MNIST


def load_CIFAR10():
    """
    Download and load the CIFAR10 dataset using torchvision.

    This function returns two dataset objects (train and test), with images
    converted to torch tensors using the `ToTensor()` transform.

    Notes:
    - The datasets are downloaded to `DATA_DIR / "train_CIFAR10"` and
      `DATA_DIR / "test_CIFAR10"`.
    - The returned objects are instances of `torchvision.datasets.CIFAR10`.
    - Data is not preloaded into memory. Images are converted to tensors only
      when accessed via indexing (e.g., dataset[i]).
    - CIFAR10 images need to be explicitly converted to torch.Tensor before
      being passed to get_batches.

    Returns:
        tuple: (train_dataset, test_dataset)
            train_dataset (torchvision.datasets.CIFAR10): Training set.
            test_dataset (torchvision.datasets.CIFAR10): Test set.
    """
    train_CIFAR10 = datasets.CIFAR10(
        root = DATA_DIR / "train_CIFAR10",
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_CIFAR10 = datasets.CIFAR10(
        root = DATA_DIR / "test_CIFAR10",
        train = False,
        transform = ToTensor(),
        download = True
    )

    return train_CIFAR10, test_CIFAR10

