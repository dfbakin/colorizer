import os

import torch
import torch.nn as nn
import torch.optim as optim

from losses import GANLoss
from models.gan_descriminator import PatchDiscriminator
from models.unet_model import UNet


def init_model(model, device):
    model.to(device)
    return model


class UNetGANWrap(nn.Module):
    def __init__(
        self,
        net_G=None,
        lr_G=2e-4,
        lr_D=2e-4,
        beta1=0.5,
        beta2=0.999,
        lambda_L1=100.0,
        device="cpu",
    ):
        super().__init__()

        self.device = device
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(
                UNet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device
            )
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(
            PatchDiscriminator(input_chn=3, n_down=3, num_filters=64), self.device
        )
        self.GANcriterion = GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

        self.loss_G = 0.0
        self.loss_D = 0.0

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data[0].to(self.device)
        self.ab = data[1].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

        return {
            "loss_G": self.loss_G.item(),
            "loss_D": self.loss_D.item(),
        }

    def test(self):
        self.net_G.eval()
        with torch.no_grad():
            self.forward()
        return {
            "fake_color": self.fake_color,
            "ab": self.ab,
        }

    def eval_generator(self, batch):
        self.net_G.eval()
        with torch.no_grad():
            self.setup_input(batch)
            self.forward()
            fake_color = self.fake_color
        return fake_color

    def save(self, path, epoch=None):
        if epoch is not None:
            path = path + "/epoch_" + str(epoch)
        else:
            path = path + "/latest"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net_G.state_dict(), path + "/net_G.pth")
        torch.save(self.net_D.state_dict(), path + "/net_D.pth")
        torch.save(self.opt_G.state_dict(), path + "/opt_G.pth")
        torch.save(self.opt_D.state_dict(), path + "/opt_D.pth")

    def load(self, path):
        self.net_G.load_state_dict(torch.load(path + "/net_G.pth"))
        self.net_D.load_state_dict(torch.load(path + "/net_D.pth"))
        self.opt_G.load_state_dict(torch.load(path + "/opt_G.pth"))
        self.opt_D.load_state_dict(torch.load(path + "/opt_D.pth"))
        self.net_G.eval()
        self.net_D.eval()
        self.opt_G.eval()
        self.opt_D.eval()
        return self.net_G, self.net_D, self.opt_G, self.opt_D
