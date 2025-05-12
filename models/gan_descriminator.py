import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, input_chn, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_chn, num_filters, norm=False)]
        for i in range(n_down):
            current_filter_num = num_filters * (2**i)
            model += [self.get_layers(current_filter_num, current_filter_num * 2)]
        model += [
            self.get_layers(
                num_filters * (2**n_down), 1, stride=1, norm=False, act=False
            )
        ]

        self.model = nn.Sequential(*model)

    def get_layers(
        self,
        in_chn,
        filter_num,
        kernel_size=4,
        stride=2,
        padding=1,
        norm=True,
        act=True,
    ):
        layers = [
            nn.Conv2d(in_chn, filter_num, kernel_size, stride, padding, bias=(not norm))
        ]
        if norm:
            layers += [nn.BatchNorm2d(filter_num)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
