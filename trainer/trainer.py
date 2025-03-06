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

    def train(self, epoch_n):
        for epoch in range(epoch_n):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm.tqdm(self.train_dataloader):
            loss = self._proccess_batch(batch, training=True)
            running_loss += loss

        epoch_loss = running_loss / len(self.train_dataloader)
        print(f"[Train: {epoch + 1}] loss: {epoch_loss:.3f}")

        if self.val_dataloader is not None:
            self._validate_epoch(epoch)

    def _validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        for batch in tqdm.tqdm(self.val_dataloader):
            loss = self._proccess_batch(batch, training=False)
            running_loss += loss

        epoch_loss = running_loss / len(self.val_dataloader)
        print(f"[Validation: {epoch + 1}] loss: {epoch_loss:.3f}")

    def _proccess_batch(self, batch, training=True):
        L_channel, ab_channels = self.batch2device(batch, self.device)

        if training:
            self.optimizer.zero_grad()
        outputs = self.model(L_channel)
        # print(outputs.shape)
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
