from torch.nn.functional import conv2d
import torch

from myproject.utils.utils import ensure_conv_input, Parameter

class Dense:
    """
    Fully connected layer (linear transformation).

    Attributes:
        input_size (int): Number of input features.
        output_size (int): Number of output neurons.
        weights (torch.Tensor): Weight matrix of shape (output_size, input_size).
        bias (torch.Tensor): Bias vector of shape (1, output_size).
    """
    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = Parameter(
            torch.randn(output_size, input_size) * 0.01
        )
        self.bias = Parameter(
            torch.zeros(output_size) 
        )

    def parameters(self):
        return [self.weights, self.bias]
    
    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.output = self.input @ self.weights.data.T + self.bias.data

        return self.output

    def backwards(self, grad_output):
        '''
        Computes the gradient of the loss function w.r.t weights, bias and input
        of the Dense layer. The gradients w.r.t weights and the bias are stored
        as attributes of self.weights and self.bias. 

        The method returns the gradient of the loss function w.r.t input or the layer.
        '''
        self.weights.grad = grad_output.T @ self.input / self.batch_size
        self.bias.grad = grad_output.sum(dim=0)  / self.batch_size

        self.grad_input = grad_output @ self.weights.data

        return self.grad_input

    def get_config(self):
        return {
            "type": "Dense",
            "output_size": int(self.output_size),
            "input_size": int(self.input_size) 
        }
    
    def state_dict(self):
        return {
            "weights": self.weights.data,
            "bias": self.bias.data
        }
    
    def load_state_dict(self, state):
        self.weights.data = state["weights"]
        self.bias.data = state["bias"]


    def stats(self):
        '''
        Returns a list with two dictionaries, one for the weights
        and one for the bias. They both contain the mean, std, max
        and min values of these parameters 
        '''
        weights_stats = {
            "mean": self.weights.data.mean().item(),
            "std": self.weights.data.std().item(),
            "max": self.weights.data.max().item(),
            "min": self.weights.data.min().item()
        }

        bias_stats = {
            "mean": self.bias.data.mean().item(),
            "std": self.bias.data.std().item(),
            "max": self.bias.data.max().item(),
            "min": self.bias.data.min().item()
        }

        return weights_stats, bias_stats

class ReLU:
    """
    Rectified Linear Unit activation function.

    The layer outputs max(0, x) element-wise.  
    During the forward pass, a mask is stored to enable efficient backward pass.

    Attributes:
        mask (torch.Tensor): Boolean tensor indicating where input > 0.
    """
    def forward(self, input):
        self.mask = input > 0.0
        return self.mask * input
    
    def backwards(self, grad_output):
        return grad_output * self.mask

    def get_config(self):
        return {
            "type": "ReLU"
        }


class Tanh:
    """
    Hyperbolic tangent activation function.

    During forward pass, the output is stored to compute gradients efficiently
    in the backward pass.

    Attributes:
        output (torch.Tensor): Stores tanh(x) for use in backward.
    """
    def forward(self, input):
        self.output = torch.tanh(input)
        return self.output
    
    def backwards(self, grad_output):
        grad_input = grad_output * (1 - self.output**2)
        return grad_input

    def get_config(self):
        return {
            "type": "Tanh"
        }
    

class ConvolutionalLayer:
    def __init__(self, output_channels, input_channels, kernel_size, stride = 1, padding = 0):
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.weights = Parameter(
            torch.randn(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size) 
        ) 
        self.bias = Parameter(
            torch.zeros(self.output_channels)
        ) 

    def parameters(self):
        return [self.weights, self.bias]

    def forward(self, input): 
        """
        Forward pass for convolution.

        Input shape: (batch, in_channels, H, W)
        Output shape: (batch, out_channels, H_out, W_out)

        Uses custom conv2d implementation.

        Stores input and computed output for backprop.
        """
        self.input = ensure_conv_input(input)
        self.input_size = input.shape[2]
        self.batch_size = self.input.shape[0]

        self.output = conv2d(
            self.input,
            self.weights.data,
            self.bias.data,
            self.stride,
            self.padding
        )

        return self.output
    
    def backwards(self, grad_output):
        """
        Backprop for convolution.

        grad_output shape: (batch, out_channels, H_out, W_out)

        Computes:
        - grad_bias: gradient wrt bias
        - grad_weights: gradient wrt filters
        - grad_input: gradient wrt input (propagated to previous layer)

        Notes:
        - Weight gradient computed using convolution between input and grad_output.
        - Input gradient computed using flipped weights (standard conv backprop).
        """
        self.bias.grad = grad_output.sum(dim = (0, 2, 3))

        input_permuted = self.input.permute(1,0,2,3)
        grad_output_permuted = grad_output.permute(1,0,2,3)

        self.weights.grad = conv2d(
            input_permuted,
            grad_output_permuted
        ).permute(1,0,2,3)

        '''
        DEPRECATED PASS
        self.grad_weights = torch.zeros(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)

        reduced_input = self.input.mean(0)
        reduced_grad_output = grad_output.mean(0)

        for i in range(self.output_channels):
            for j in range(self.input_channels):
                self.grad_weights[i,j] = conv2d(
                    reduced_input[j].unsqueeze(0).unsqueeze(0),
                    reduced_grad_output[i].unsqueeze(0).unsqueeze(0),
                    stride = self.stride,
                    padding = self.padding
                )
        '''

        weights_flipped = self.weights.data.flip(dims = [-1, -2]).permute(1,0,2,3)
        
        self.grad_input = conv2d(
            grad_output,
            weights_flipped,
            padding = self.kernel_size - 1
        )

        '''
        DEPRECATED PASS
        self.grad_input = torch.zeros(self.batch_size, self.input_channels, self.input_size, self.input_size)

        for n in range(self.batch_size):
            for j in range(self.input_channels):
                self.grad_input[n,j] = conv2d(
                    grad_output[n].unsqueeze(0),
                    weights_flipped[j].unsqueeze(0),
                    stride = self.stride,
                    padding = self.kernel_size - 1
                ) 
        '''
        return self.grad_input

    def get_config(self):
        return {
            "type": "ConvolutionalLayer",
            "output_channels": int(self.output_channels),
            "input_channels": int(self.input_channels),
            "kernel_size": int(self.kernel_size)
        }
    
    def state_dict(self):
        return {
            "weights": self.weights.data,
            "bias": self.bias.data
        }
        
    def load_state_dict(self, state):
        self.weights.data = state["weights"]
        self.bias.data = state["bias"]

    
    def stats(self):
        '''
        Returns a list of dictionaries with the mean, std, max and min
        values of the weights and bias of the layer. .item() assures 
        values are JSONserializable.
        '''
        weights_stats = {
            "mean": self.weights.data.mean().item(),
            "std": self.weights.data.std().item(),
            "max": self.weights.data.max().item(),
            "min": self.weights.data.min().item()
        }

        bias_stats = {
            "mean": self.bias.data.mean().item(),
            "std": self.bias.data.std().item(),
            "max": self.bias.data.max().item(),
            "min": self.bias.data.min().item()
        }

        return weights_stats, bias_stats


class ReshapeLayer:
    """
    Flatten layer used to bridge convolutional layers and fully connected layers.

    Converts:
        (batch, C, H, W) → (batch, C * H * W)

    During backprop, reshapes gradients back to the original input shape.
    """
    def forward(self, x):
        # Save original shape so we can restore it during backprop
        self.input_shape = x.shape

        # Flatten all dimensions except batch
        self.output_shape = x.reshape(x.size(0), -1)
        return self.output_shape

    def backwards(self, grad_output):
        # Restore gradient to the shape expected by the previous layer
        return grad_output.reshape(self.input_shape)

    def get_config(self):
        # No learnable parameters; config is purely structural
        return {
            "type": "ReshapeLayer"
        }


class MaxPool:
    """
    2D Max Pooling layer (no learnable parameters).

    Uses torch.unfold to extract sliding windows and applies max
    over each window. Backprop routes gradients only to max locations.
    """
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        # Default stride = kernel size (standard non-overlapping pooling)
        self.stride = stride if stride is not None else kernel_size

    def forward(self, input):
        """
        Forward pass.

        input shape:  (N, C, H, W)
        output shape: (N, C, out_h, out_w)

        Stores argmax indices for use during backprop.
        """
        self.input_shape = input.shape
        k = self.kernel_size
        s = self.stride

        # Extract sliding k×k windows using unfold
        # Result shape: (N, C, out_h, out_w, k, k)
        input_unfold = input.unfold(2, k, s).unfold(3, k, s)

        # Flatten each k×k window to length k*k
        input_unfold_flat = input_unfold.reshape(
            *input_unfold.shape[:-2], k * k
        )

        # Save index of max value in each window
        self.argmax = input_unfold_flat.argmax(dim=-1)

        # Max pooling over last dimension (k*k)
        output = input_unfold_flat.max(dim=-1).values
        return output

    def backwards(self, grad_output):
        """
        Backward pass.

        grad_output shape: (N, C, out_h, out_w)

        Gradient is routed only to the max element of each pooling window.
        """
        N, C, H, W = self.input_shape
        k = self.kernel_size
        s = self.stride
        out_h, out_w = grad_output.shape[2:]

        # Initialize gradient wrt input with zeros
        grad_input = torch.zeros((N, C, H, W))

        # Unfold grad_input to match forward window structure
        grad_input_unfold = grad_input.unfold(2, k, s).unfold(3, k, s)
        grad_input_unfold = grad_input_unfold.reshape(
            N, C, out_h, out_w, k * k
        )

        # Scatter gradients back to max locations only
        grad_input_unfold.scatter_(
            dim=-1,
            index=self.argmax.unsqueeze(-1),
            src=grad_output.unsqueeze(-1)
        )

        return grad_input
    
    def get_config(self):
        return {
            "type": "MaxPool",
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }


class GAP:
    """
    Global Average Pooling (GAP) layer.

    Replaces fully connected layers at the end of CNNs by averaging
    each feature map into a single value.

    Input shape:  (batch, channels, H, W)
    Output shape: (batch, channels)
    """
    def forward(self, input):
        # Save input for shape reference during backprop
        self.input = input

        # Average over spatial dimensions (H, W)
        output = input.mean(dim=(-1, -2))
        return output

    def backwards(self, grad_output):
        """
        Backward pass.

        grad_output shape: (batch, channels)

        Each spatial location contributed equally to the mean,
        so the gradient is distributed uniformly over H*W.
        """
        B, C, H, W = self.input.shape

        # Expand gradient back to spatial dimensions and normalize
        grad_input = grad_output[:, :, None, None] / (H * W)

        # Broadcasting fills (H, W)
        grad_input = grad_input.expand(B, C, H, W)
        return grad_input

    def get_config(self):
        # No learnable parameters
        return {
            "type": "GAP"
        }


        
