from myproject.model.activations import Dense, ReLU, Tanh, ConvolutionalLayer, MaxPool, GAP
from myproject.loss.loss_functions import CrossEntropy
from myproject.model.neural_network import NeuralNetwork
from myproject.data_scripts.data_preprocessing import process_MNIST, process_CIFAR10
from myproject.optimizer.optimizer import SGD, SGDWithMomentum
from myproject.model.activations import ReshapeLayer
from myproject.training.trainer import Trainer
from myproject.utils.io import Logger
from myproject.scheduler.scheduler import ReducelrOnPlateau
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected device: {device}")

train_images, train_targets, test_images, test_targets = process_MNIST(device = device)

# Inititalize model. 
model = NeuralNetwork([
    ConvolutionalLayer(8, 1, 3, padding=1),
    Tanh(),
    ConvolutionalLayer(8, 8, 3, padding = 1),
    Tanh(),
    MaxPool(2,2),
    ConvolutionalLayer(16, 8, 3, padding = 1),
    Tanh(),
    ConvolutionalLayer(8, 16, 3, padding = 1),
    Tanh(),
    MaxPool(2,2),
    ReshapeLayer(),
    Dense(256, 8),
    Tanh(),
    Dense(10, 256)
])

model.to(device)

optimizer = SGDWithMomentum(model.parameters(), lr = 0.01)
loss_fn = CrossEntropy()
logger = Logger()

scheduler = ReducelrOnPlateau(
    optimizer,
    patience = 2,
    factor = 0.5,
    min_lr = 1e-5,
    threshold = 1e-4
)

trainer = Trainer(
    model,
    loss_fn,
    logger,
    scheduler,
)

trainer.train_model(
    optimizer = optimizer,
    train_data = train_images,
    train_targets = train_targets,
    eval_data = test_images,
    eval_targets = test_targets,
    num_epochs = 1,
    batch_size = 64,
    eval=True
)