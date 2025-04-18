import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
from IPython.display import clear_output

from utils import WandbLogger


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        use_logger=False,
        config=None,
        eval_quality_metric=nn.MSELoss(),
        device="cpu",
        use_checkpointing=False,
        checkpoit_dir="",
    ):
        self.model = model
        self.device = device
        self.eval_quality_metric = eval_quality_metric
        self.use_checkpointing = use_checkpointing
        self.checkpoint_dir = checkpoit_dir

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if use_logger:
            self.logger = WandbLogger("colorizer", config)
        else:
            self.logger = None

        self.train_losses = {
            "loss_G": [],
            "loss_D": [],
        }
        self.val_loss = []

        self.current_trained_epochs = 0

    def train(self, epoch_n):
        if epoch_n <= self.current_trained_epochs:
            raise ValueError(
                "The number of epochs should be greater than the current trained epochs"
            )

        for epoch in range(self.current_trained_epochs, epoch_n):
            try:
                self._train_epoch(epoch)
            except KeyboardInterrupt:
                print("Training interrupted, but current training progress is preserve")
                print(f"Current trained epochs: {self.current_trained_epochs}")
                break
            except Exception as err:
                if "out of memory" in str(err):
                    print("Out of memory, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise err
            self.current_trained_epochs = epoch + 1

            if self.logger:
                self.logger.log(
                    {"train_loss": self.train_losses.mean},
                    step=self.current_trained_epochs,
                )
                if self.val_dataloader is not None:
                    self.logger.log(
                        {"val_loss": self.val_losses.mean},
                        step=self.current_trained_epochs,
                    )
            if self.use_checkpointing:
                self.model.save(self.checkpoint_dir, epoch=self.current_trained_epochs)

    def _train_epoch(self, epoch):
        # self.model.train()
        epoch_loss = {
            "loss_G": 0.0,
            "loss_D": 0.0,
        }
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.train_dataloader),
            desc="train",
            total=len(self.train_dataloader),
        ):
            self.model.setup_input(batch)
            self.model.forward()
            losses = self.model.optimize()
            epoch_loss["loss_G"] += losses["loss_G"]
            epoch_loss["loss_D"] += losses["loss_D"]

        epoch_loss["loss_G"] /= len(self.train_dataloader)
        epoch_loss["loss_D"] /= len(self.train_dataloader)
        if self.logger:
            self.logger.log(
                {
                    "train_loss_G": epoch_loss["loss_G"],
                    "train_loss_D": epoch_loss["loss_D"],
                },
                step=self.current_trained_epochs,
            )

        # self.train_losses.update(epoch_loss)
        self.train_losses["loss_G"].append(epoch_loss["loss_G"])
        self.train_losses["loss_D"].append(epoch_loss["loss_D"])

        clear_output(wait=True)
        print(
            f"[Train: {epoch + 1}] loss_G: {epoch_loss['loss_G']:.3f} "
            f"loss_D: {epoch_loss['loss_D']:.3f}"
        )

        if self.val_dataloader is not None:
            self._validate_epoch(epoch)

        # Plot generator loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses["loss_G"], label="Train Loss G")
        plt.title("Generator Loss (with GAN loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plot discriminator loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses["loss_D"], label="Train Loss D")
        plt.title("Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        if self.val_dataloader is not None:
            plt.plot(self.val_loss, label="Validation Loss G")
        plt.title("MSE Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def _validate_epoch(self, epoch):
        running_loss = 0.0
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.val_dataloader),
            desc="val",
            total=len(self.val_dataloader),
        ):
            ab_pred = self.model.eval_generator(batch)
            running_loss += self.eval_quality_metric(
                ab_pred, batch[1].to(self.device)
            ).item()

        epoch_loss = running_loss / len(self.val_dataloader)
        self.val_loss.append(epoch_loss)
        print(f"[Validation: {epoch + 1}] loss_G: {epoch_loss:.3f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        return self.model
