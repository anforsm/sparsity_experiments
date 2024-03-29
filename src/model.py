import torch
from unet import UNet, ConvPass


class Model(torch.nn.Module):

    def __init__(
            self,
            in_channels=1,
            output_shapes=[3],
            fmap_inc_factor=5,
            downsample_factors=((1,2,2),(1,2,2),(2,2,2)),
            kernel_size_down = (
                ((3, 3, 3), (3, 3, 3)),
                ((3, 3, 3), (3, 3, 3)),
                ((3, 3, 3), (3, 3, 3)),
                ((1, 3, 3), (1, 3, 3))
            ),
            kernel_size_up = (
                ((3, 3, 3), (3, 3, 3)),
                ((3, 3, 3), (3, 3, 3)),
                ((3, 3, 3), (3, 3, 3))
            )
    ):

        super().__init__()

        num_fmaps = 12

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True,
                num_heads=1)

        self.aff_head = ConvPass(num_fmaps, output_shapes[0], [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)
        affs = self.aff_head(z)

        return affs


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss

    def forward(
            self,
            affs_prediction,
            affs_target,
            affs_weights):

        loss = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return loss
