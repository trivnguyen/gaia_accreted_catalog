
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    MLP with a variable number of hidden layers.

    Attributes
    ----------
    layers : nn.ModuleList
        The layers of the MLP.
    activation_fn : callable
        The activation function to use.
    """
    def __init__(self, input_size, output_size, hidden_sizes=[512],
                 activation_fn=nn.ReLU(), transfer_layers=None):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        output_size : int
            The number of classes
        hidden_sizes : list of int, optional
            The sizes of the hidden layers. Default: [512]
        activation_fn : callable, optional
            The activation function to use. Default: nn.ReLU()
        transfer_layers: list of int, optional
            The layers to transfer (freeze). Default: None
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn
        self.transfer_layers = transfer_layers

        # Create a list of all layer sizes: input, hidden, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Store the activation function
        self.activation_fn = activation_fn

        # Freeze layers if transfer is specified
        self.freeze_layers(transfer_layers)

    def freeze_layers(self, transfer_layers):
        """Freeze specified layers by setting requires_grad to False."""
        if transfer_layers is not None:
            for i, layer in enumerate(self.layers):
                if i in transfer_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, x):
        # Apply layers and activation function
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function to all but last layer
                x = self.activation_fn(x)
        return x
