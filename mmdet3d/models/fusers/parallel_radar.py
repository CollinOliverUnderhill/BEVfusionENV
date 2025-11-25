from typing import List, Sequence

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ParallelRadarFuser"]


@FUSERS.register_module()
class ParallelRadarFuser(nn.Module):
    """Fuse radar features via a parallel branch.

    This module keeps the original BEV features (camera/lidar/etc.) fused by a
    Conv2d block while processing radar features with an independent block. The
    two branches are later aligned to the same number of channels and summed to
    form the final BEV representation. This design allows radar features to be
    injected without forcing them to share the same projection weights as the
    other modalities.

    Args:
        primary_in_channels (Sequence[int]): Channel dimensions of all
            non-radar features. They will be concatenated before the primary
            Conv2d block.
        radar_in_channels (int): Channel dimension of the radar feature map.
        out_channels (int): Output channel dimension of the fused BEV feature.
        radar_channels (int, optional): Intermediate channels produced by the
            radar branch before alignment. Defaults to ``out_channels``.
        radar_index (int, optional): Index of the radar tensor in the incoming
            feature list. Defaults to ``-1`` (the last element).
    """

    def __init__(
        self,
        primary_in_channels: Sequence[int],
        radar_in_channels: int,
        out_channels: int,
        radar_channels: int = None,
        radar_index: int = -1,
    ) -> None:
        super().__init__()
        if len(primary_in_channels) == 0:
            raise ValueError("primary_in_channels must contain at least one entry")
        self.radar_index = radar_index
        self.primary_fuser = nn.Sequential(
            nn.Conv2d(sum(primary_in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        radar_channels = radar_channels or out_channels
        self.radar_branch = nn.Sequential(
            nn.Conv2d(radar_in_channels, radar_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(radar_channels),
            nn.ReLU(True),
        )
        if radar_channels != out_channels:
            self.radar_align = nn.Sequential(
                nn.Conv2d(radar_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.radar_align = nn.Identity()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if len(inputs) == 0:
            raise ValueError("ParallelRadarFuser expects at least one input tensor")

        # Resolve the radar feature index (supports negative indexing just like Python).
        radar_index = self.radar_index if self.radar_index >= 0 else len(inputs) + self.radar_index
        if radar_index < 0 or radar_index >= len(inputs):
            raise ValueError(
                f"Radar feature index {self.radar_index} is out of range for the inputs list"
            )

        radar_feature = inputs[radar_index]
        primary = [feat for idx, feat in enumerate(inputs) if idx != radar_index]
        if len(primary) == 0:
            raise ValueError("ParallelRadarFuser expects at least one non-radar feature")

        fused_primary = self.primary_fuser(torch.cat(primary, dim=1))
        fused_radar = self.radar_branch(radar_feature)
        fused_radar = self.radar_align(fused_radar)
        return fused_primary + fused_radar
