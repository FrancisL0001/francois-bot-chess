import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual          # skip connection — add BEFORE the final relu
        return F.relu(out)

class PolicyNetwork(nn.Module):
    def __init__(self, in_planes: int, num_filters: int, num_blocks: int, num_moves: int):
        super().__init__()
        # Stem: lift 17 input planes up to num_filters, stay 8x8
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_blocks)])

        # Policy head: 1x1 conv to a few planes, flatten, project to move logits
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, num_moves)

    def forward(self, x):                 # x: (B, 17, 8, 8)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.policy_conv(x)           # (B, 32, 8, 8)
        x = x.flatten(1)                  # (B, 32*8*8)
        return self.policy_fc(x)          # (B, num_moves) — RAW logits, no softmax
    
