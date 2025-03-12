import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class DirectionalConv(nn.Module):
    """
    Direction-aware convolution for enhancing horizontal structures like staff lines.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(1, 9), stride=1, padding=None, dilation=1):
        super(DirectionalConv, self).__init__()
        if padding is None:
            # Calculate padding to maintain input size
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        # Initialize with horizontal edge detection bias
        nn.init.constant_(self.conv.bias, 0.0)
        nn.init.xavier_uniform_(self.conv.weight)
        
    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    """
    Channel attention module to emphasize important features across channels.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important regions of the image.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        return torch.sigmoid(out) * x


class StaffLineConstraintModule(nn.Module):
    """
    A module that enforces musical staff line structure knowledge.
    """
    def __init__(self, in_channels, num_staff_lines=5):
        super(StaffLineConstraintModule, self).__init__()
        self.num_staff_lines = num_staff_lines
        
        # Feature transformation
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Vertical spacing pattern encoder 
        self.spacing_encoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(15, 1), padding=(7, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=(5, 1), padding=(2, 0)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = F.relu(self.conv1(x))
        features = F.relu(self.conv2(features))
        
        # Generate a weight map that enhances regions with staff-like spacing
        spacing_weights = self.spacing_encoder(features)
        
        # Apply the weighting
        enhanced = x * spacing_weights
        
        return enhanced


class DoubleConv(nn.Module):
    """Double convolution block."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block."""
    def __init__(self, in_channels, out_channels, use_directional=False):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        
        if use_directional:
            self.conv = nn.Sequential(
                DirectionalConv(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upsampling block."""
    def __init__(self, in_channels, out_channels, use_attention=False, use_directional=False):
        super(Up, self).__init__()
        self.use_attention = use_attention
        
        # Upsampling with transposed convolution
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # In the upsampling block, the input will have (in_channels // 2) + out_channels channels
        # after concatenation with the skip connection
        if use_directional:
            self.conv = nn.Sequential(
                DirectionalConv(in_channels // 2 + out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)
            
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)
            self.spatial_attention = SpatialAttention()
        
    def forward(self, x1, x2):
        # x1 is the upsampled feature, x2 is the skip connection
        x1 = self.up(x1)
        
        # Padding in case the dimensions don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        # Process concatenated features
        x = self.conv(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            
        return x


class StaffLineDetectionNet(nn.Module):
    """
    Complete network for staff line detection.
    """
    def __init__(self, n_channels=1, n_classes=1):
        super(StaffLineDetectionNet, self).__init__()
        
        # Initial convolution
        self.inc = nn.Sequential(
            DirectionalConv(n_channels, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path
        self.down1 = Down(64, 128, use_directional=True)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Bottleneck
        self.neck = nn.Sequential(
            DirectionalConv(1024, 1024, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path
        self.up1 = Up(1024, 512, use_attention=True)
        self.up2 = Up(512, 256, use_attention=True)
        self.up3 = Up(256, 128, use_attention=True, use_directional=True)
        self.up4 = Up(128, 64, use_directional=True)
        
        # Staff line constraint module
        self.staff_constraint = StaffLineConstraintModule(64, num_staff_lines=5)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path with skip connections
        x1 = self.inc(x)           # 64 channels
        x2 = self.down1(x1)        # 128 channels
        x3 = self.down2(x2)        # 256 channels
        x4 = self.down3(x3)        # 512 channels
        x5 = self.down4(x4)        # 1024 channels
        
        # Bottleneck
        x5 = self.neck(x5)         # 1024 channels
        
        # Decoder path
        x = self.up1(x5, x4)       # 512 channels
        x = self.up2(x, x3)        # 256 channels
        x = self.up3(x, x2)        # 128 channels
        x = self.up4(x, x1)        # 64 channels
        
        # Apply staff line constraints
        x = self.staff_constraint(x)
        
        # Output segmentation map
        logits = self.outc(x)
        
        return logits


def staff_line_dice_loss(pred, target, smooth=1.0):
    """
    Specialized Dice loss for staff line segmentation.
    Gives higher weight to horizontal continuity.
    """
    # Apply horizontal gradient to both prediction and target
    h_grad_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    h_grad_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # Standard dice
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Horizontal continuity term - penalize predicted edges that don't exist in target
    h_grad_pred_flat = h_grad_pred.view(-1)
    h_grad_target_flat = h_grad_target.view(-1)
    
    continuity_penalty = (h_grad_pred_flat * (1 - h_grad_target_flat)).mean()
    
    # Combine losses
    return 1 - dice + 0.2 * continuity_penalty


def post_process_staff_lines(pred, min_length=50, max_gap=10):
    """
    Post-process predicted staff lines to improve connectivity.
    """
    # Check if pred is a tensor, convert to numpy
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy().squeeze()
    else:
        pred_np = pred
    
    # Threshold
    binary = (pred_np > 0.5).astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = np.ones((1, max_gap), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Use HoughLinesP to find and enhance lines
    lines = cv2.HoughLinesP(
        closed, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20, 
        minLineLength=min_length, 
        maxLineGap=max_gap
    )
    
    # Create a mask from detected lines
    line_mask = np.zeros_like(binary)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Combine with original prediction
    enhanced = np.maximum(closed, line_mask)
    
    # Convert back to tensor if input was tensor
    if isinstance(pred, torch.Tensor):
        return torch.from_numpy(enhanced).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        return enhanced / 255.0