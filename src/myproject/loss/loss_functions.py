import torch

class CrossEntropy():
    """
    CrossEntropy loss module.

    This class implements a standard cross-entropy loss for multi-class
    classification using logits as input.

    Methods
    -------
    forward(logits, output)
        Computes the average cross-entropy loss over the batch.
        - logits: raw model outputs (unnormalized scores) of shape (batch_size, number_of_classes)
        - output: ground-truth labels of shape (batch_size,)
        Returns:
        - scalar loss value (mean over batch)

    backwards(output)
        Computes the gradient of the loss with respect to the input logits.
        The gradient is derived from the softmax + cross-entropy formulation:
        dL/dz = softmax(z) - y
        where y is the one-hot encoded label vector.

        The method reuses cached softmax log-probabilities computed during
        forward pass for efficiency.

    Notes
    -----
    - Assumes `forward` is called before `backwards`.
    - Uses `torch.log_softmax` for numerical stability.
    - Gradient is returned as a tensor of shape (batch_size, number_of_classes).
    """
    def forward(self, logits, y_true):
        self.y_true = y_true
        self.batch_size = y_true.shape[0]
        self.log_probs = torch.log_softmax(logits, dim = 1)

        avg_batch_CE = -self.log_probs[torch.arange(self.batch_size), self.y_true].mean()
        return avg_batch_CE
    
    def backwards(self, output):
        predicted_probs = torch.exp(self.log_probs)
        
        grad_output = predicted_probs.clone()
        grad_output[torch.arange(self.batch_size), output] -= 1
        return grad_output


