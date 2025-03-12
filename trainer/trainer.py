import matplotlib.pyplot as plt
import torch
import tqdm
from IPython.display import clear_output
from utils import WandbLogger, MetricTracker

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader=None,
        scheduler=None,
        use_logger=False,
        config=None,
        device="cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if use_logger:
            self.logger = WandbLogger("colorizer", config)
        else:
            self.logger = None
        self.scheduler = scheduler

        self.train_losses = MetricTracker("train_loss")
        self.val_losses = MetricTracker("val_loss")

        self.current_trained_epochs = 0


    @staticmethod
    def batch2device(batch, device):
        return [item.to(device) for item in batch]

    def train(self, epoch_n):
        if epoch_n <= self.current_trained_epochs:
            raise ValueError("The number of epochs should be greater than the current trained epochs")

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
            self.logger.log({"train_loss": self.train_losses.mean}, 
                            step=self.current_trained_epochs)
            if self.val_dataloader is not None:
                self.logger.log({"val_loss": self.val_losses.mean}, 
                                step=self.current_trained_epochs)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc="train", 
                                          total=len(self.train_dataloader)):
            
            loss = self._proccess_batch(batch, training=True)
            running_loss += loss
        
        epoch_loss = running_loss / len(self.train_dataloader)
        self.train_losses.update(epoch_loss)

        clear_output(wait=True)
        print(f"[Train: {epoch + 1}] loss: {epoch_loss:.3f}")

        if self.val_dataloader is not None:
            self._validate_epoch(epoch)
        
        # Plot and clear output
        plt.plot(self.train_losses.values, label="Train Loss")
        if self.val_dataloader is not None:
            plt.plot(self.val_losses.values, label="Validation Loss")
        plt.legend()
        plt.show()

    def _validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in tqdm.tqdm(enumerate(self.val_dataloader), desc="val", 
                                          total=len(self.val_dataloader)):
                loss = self._proccess_batch(batch, training=False)
                running_loss += loss

        epoch_loss = running_loss / len(self.val_dataloader)
        self.val_losses.update(epoch_loss)
        print(f"[Validation: {epoch + 1}] loss: {epoch_loss:.3f}")

    def _proccess_batch(self, batch, training=True):
        L_channel, ab_channels = self.batch2device(batch, self.device)

        if training:
            self.optimizer.zero_grad()
        outputs = self.model(L_channel)
        loss = torch.mean(self.criterion(outputs, ab_channels))
        if training:
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        return self.model
