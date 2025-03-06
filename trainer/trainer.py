import torch
import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader=None,
        scheduler=None,
        logger=None,
        device="cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.scheduler = scheduler

    @staticmethod
    def batch2device(batch, device):
        return [item.to(device) for item in batch]

    def train(self, train_loader, epoch_n):
        for epoch in range(epoch_n):
            self._train_epoch(train_loader, epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for i, batch in tqdm.tqdm(enumerate(self.train_loader)):
            loss = self._proccess_batch(batch, training=True)
            running_loss += loss

        print(f"[Train: {epoch + 1}, {i + 1}] loss: {running_loss / i:.3f}")

        if self.val_dataloader is not None:
            self._validate_epoch(epoch)

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        for i, batch in tqdm.tqdm(enumerate(self.val_dataloader)):
            loss = self._proccess_batch(batch, training=False)
            running_loss += loss

        print(f"[Validation: {epoch + 1}, {i + 1}] loss: {running_loss / i:.3f}")

    def _proccess_batch(self, batch, training=True):
        L_channel, ab_channels = self.batch2device(batch, self.device)

        if training:
            self.optimizer.zero_grad()
        outputs = self.model(L_channel)
        loss = self.criterion(outputs, ab_channels)
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
