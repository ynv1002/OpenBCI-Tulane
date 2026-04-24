import torch
import torch.nn as nn

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        with torch.no_grad():
            norms = torch.norm(self.weight, dim=2, keepdim=True).norm(dim=3, keepdim=True)
            desired = torch.clamp(norms, 0, self.max_norm)
            self.weight *= (desired / (1e-8 + norms))
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        with torch.no_grad():
            norms = torch.norm(self.weight, dim=1, keepdim=True)
            desired = torch.clamp(norms, 0, self.max_norm)
            self.weight *= (desired / (1e-8 + norms))
        return super(LinearWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    """
    EEGNet-4,2 architecture for BCI decoding.
    Reference: Lawhern et al., 2018. "EEGNet: a compact convolutional neural network
    for EEG-based brain-computer interfaces".
    """
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 4,
        D: int = 2,
        F2: int = 8,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1
        # Pad 'same' equivalent for Conv2d.
        # Output shape: (batch, F1, n_channels, n_times)
        pad = kernel_length // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, kernel_length), padding=(0, pad), bias=False),
            nn.BatchNorm2d(self.F1),
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (n_channels, 1), groups=self.F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 2
        # Separable convolution: Depthwise time convolution -> Pointwise channel convolution
        # Output shape: (batch, F2, 1, n_times // 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding=(0, 16 // 2), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Dynamic dimension calculation for the fully connected layer
        dummy_input = torch.zeros(1, 1, n_channels, n_times)
        with torch.no_grad():
            x = self.block1(dummy_input)
            x = self.block2(x)
            self.flat_dim = x.numel()

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(self.flat_dim, n_classes, max_norm=0.25)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skorch natively passes X as (batch, channels, times). 
        # EEGNet needs an explicitly 2D-spatial dimension: (batch, 1, channels, times)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
