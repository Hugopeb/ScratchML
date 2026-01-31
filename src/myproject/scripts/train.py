from myproject.model.activations import Dense, ReLU, Tanh, ConvolutionalLayer, MaxPool, GAP
from myproject.loss.loss_functions import CrossEntropy
from myproject.model.neural_network import NeuralNetwork
from myproject.data_scripts.data_preprocessing import process_MNIST, process_CIFAR10
from myproject.optimizer.optimizer import SGD, SGDWithMomentum
from myproject.model.activations import ReshapeLayer
from myproject.training.trainer import Trainer
from myproject.utils.io import Logger

train_images, train_targets, test_images, test_targets = process_MNIST(ConvolutionalLayer=True)

# Inititalize model. 
model = NeuralNetwork([
    ConvolutionalLayer(16, 1, 5),
    Tanh(),
    ConvolutionalLayer(8, 16, 5),
    MaxPool(2, 2),
    ReshapeLayer(),
    Dense(512, 800),
    Tanh(),
    Dense(10, 512),
])

trainer = Trainer(
    model = model,
    logger = Logger(),
    loss_fn = CrossEntropy(),
)

optimizer = SGDWithMomentum(model.parameters(), lr = 0.01, momentum = 0.6)

trainer.train_model(
    optimizer = optimizer,
    train_data = train_images,
    train_targets = train_targets,
    eval_data = test_images,
    eval_targets = test_targets,
    num_epochs = 6,
    batch_size = 64,
    eval=True
)