# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)

class SpectrogramCropAugmentation(nn.Module):
    """Augmentation block that performs random cropping on spectrograms.
    
    For sEMG data, this simulates slight variations in electrode positioning
    and helps make the model more robust to spatial shifts in the input signal.
    
    Args:
        crop_size (int): Size of the frequency crop window
        crop_prob (float): Probability of applying a crop during training
        electrode_dim (int): Dimension of electrodes in the input (default: 3)
        freq_dim (int): Dimension of frequency bins in the input (default: 4)
        training_only (bool): Whether to only apply augmentation during training (default: True)
    """
    def __init__(self, crop_size: int, crop_prob: float = 0.5, 
                 electrode_dim: int = 3, freq_dim: int = 4, training_only: bool = True):
        super().__init__()
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        self.electrode_dim = electrode_dim
        self.freq_dim = freq_dim
        self.training_only = training_only
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only apply augmentation during training if training_only=True
        if self.training_only and not self.training:
            return x
            
        # Apply random cropping with probability crop_prob
        if torch.rand(1).item() > self.crop_prob:
            return x
            
        # Get input shape
        shape = x.shape
        
        # Ensure we have enough frequency bins to crop
        freq_size = shape[self.freq_dim]
        if freq_size <= self.crop_size:
            return x
            
        # Calculate max start position for cropping
        max_start = freq_size - self.crop_size
        start_idx = torch.randint(0, max_start + 1, (1,)).item()
        end_idx = start_idx + self.crop_size
        
        # Create slices for all dimensions
        slices = [slice(None)] * len(shape)
        slices[self.freq_dim] = slice(start_idx, end_idx)
        
        # Apply cropping
        x_cropped = x[tuple(slices)]
        
        # Resize back to original size using interpolation
        if self.freq_dim == 4:  # Assuming shape is (T, N, bands, electrodes, freq)
            # Reshape for F.interpolate
            orig_shape = x_cropped.shape
            x_reshaped = x_cropped.reshape(-1, 1, self.crop_size)
            x_interp = F.interpolate(x_reshaped, size=freq_size, mode='linear')
            x_cropped = x_interp.reshape(orig_shape[0], orig_shape[1], 
                                         orig_shape[2], orig_shape[3], freq_size)
        else:
            # Handle other dimension configurations if needed
            raise ValueError(f"Unsupported frequency dimension: {self.freq_dim}")
            
        return x_cropped

class RandomChannelBlockingAugmentation(nn.Module):
    """Augmentation block that randomly blocks (zeros) sEMG channels.
    
    This simulates electrode disconnections, poor contact, or signal loss during recording,
    making the model more robust to missing or corrupted electrode channels.
    
    Args:
        block_prob (float): Probability of applying channel blocking during training
        max_channels_blocked (int): Maximum number of channels that can be blocked at once
        electrode_dim (int): Dimension of electrodes in the input (default: 3)
        training_only (bool): Whether to only apply augmentation during training (default: True)
    """
    def __init__(self, block_prob: float = 0.5, max_channels_blocked: int = 4,
                 electrode_dim: int = 3, training_only: bool = True):
        super().__init__()
        self.block_prob = block_prob
        self.max_channels_blocked = max_channels_blocked
        self.electrode_dim = electrode_dim
        self.training_only = training_only
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only apply augmentation during training if training_only=True
        if self.training_only and not self.training:
            return x
            
        # Apply random channel blocking with probability block_prob
        if torch.rand(1).item() > self.block_prob:
            return x
            
        # Get input shape and number of electrode channels
        shape = x.shape
        num_channels = shape[self.electrode_dim]
        
        # Determine how many channels to block (between 1 and max_channels_blocked)
        num_to_block = torch.randint(1, min(self.max_channels_blocked + 1, num_channels), (1,)).item()
        
        # Randomly select which channels to block
        channels_to_block = torch.randperm(num_channels)[:num_to_block]
        
        # Create a mask with 0s for blocked channels and 1s elsewhere
        mask = torch.ones(num_channels, device=x.device)
        mask[channels_to_block] = 0.0
        
        # Create slices for all dimensions to broadcast the mask properly
        slices = [None] * len(shape)
        slices[self.electrode_dim] = slice(None)
        mask = mask[tuple(slices)]
        
        # Apply mask (zeroing out blocked channels)
        x_augmented = x * mask
        
        return x_augmented

class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC

class LSTMBlock(nn.Module):
    """An LSTM block that processes sequential data.
    
    Args:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        bidirectional (bool): If True, use a bidirectional LSTM.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # (T, N, *) format expected
            bidirectional=bidirectional
        )
        self.output_size = hidden_size * (2 if bidirectional else 1)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(inputs)  # (T, N, hidden_size)
        return x

class MultiBandLSTMBlock(nn.Module):
    """An LSTM block that processes each frequency band separately.
    
    Args:
        input_size (int): Number of expected features in the input (per band).
        hidden_size (int): Number of features in the hidden state.
        num_bands (int): Number of frequency bands.
        num_layers (int): Number of recurrent layers.
        bidirectional (bool): If True, use a bidirectional LSTM.
    """
    def __init__(self, input_size: int, hidden_size: int, num_bands: int = 2, 
                 num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = 2
        
        # Create a separate LSTM for each band
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False,  # (T, N, *) format expected
                bidirectional=bidirectional
                #dropout=dropout if num_layers > 1 else 0
            ) for _ in range(num_bands)
        ])
        
        #self.layer_norms = nn.ModuleList([
          #nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
          #for _ in range(num_bands)
        #])
        
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # input shape: (T, N, num_bands, C, freq)
        T, N, bands, C, freq = inputs.shape
        
        # Reshape to process each band separately
        x_reshaped = inputs.permute(2, 0, 1, 3, 4)  # (bands, T, N, C, freq)
        x_reshaped = x_reshaped.reshape(bands, T, N, -1)  # (bands, T, N, C*freq)
        
        # Process each band with its dedicated LSTM
        outputs = []
        for band_idx in range(self.num_bands):
            band_out, _ = self.lstms[band_idx](x_reshaped[band_idx])  # (T, N, output_size)
            #band_out = self.layer_norms[band_idx](band_out)
            outputs.append(band_out)
        
        # Stack outputs back along the band dimension
        x = torch.stack(outputs, dim=2)  # (T, N, bands, output_size)
        
        return x

class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (list): A list of kernel sizes matching `block_channels`.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: Sequence[int] = (32, 32, 32, 32),
    ) -> None:
        super().__init__()

        assert len(block_channels) == len(kernel_width), "block_channels and kernel_width must have the same length"

        tds_conv_blocks: list[nn.Module] = []
        in_features = num_features  # Track changing feature size

        for i, (channels, k_width) in enumerate(zip(block_channels, kernel_width)):
            print(f"Before block {i}: in_features = {in_features}, channels = {channels}")
            assert (
                in_features % channels == 0
            ), f"block_channels must evenly divide num_features ({in_features} % {channels} != 0)"
            
            tds_conv_blocks.extend([
                TDSConv2dBlock(channels, in_features // channels, k_width),
                TDSFullyConnectedBlock(in_features),
            ])
            
            # Update in_features if needed (Optional: Adjust if feature size changes)
            in_features = channels * (in_features // channels)  # Uncomment if shape is changing

            print(f"After block {i}: in_features = {in_features}")

        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)