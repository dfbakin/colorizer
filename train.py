import torch
import torch.utils.data as torch_data

import losses
import trainer
from models import UNet
from utils import ColorizationDataset

REQUIRED_SIZE = (128, 128)
combined_dataset = torch_data.ConcatDataset(
    [
        ColorizationDataset("datasets/coco_2017_test", resize=REQUIRED_SIZE),
        ColorizationDataset(
            "datasets/imagenet_classes", resize=REQUIRED_SIZE, classes_folders=True
        ),
    ]
)
train_dataset, val_dataset, test_dataset = torch_data.random_split(
    combined_dataset, [0.05, 0.01, 0.94]
)  # debug lengths: 4534 907 85229
print(*[len(item) for item in [train_dataset, val_dataset, test_dataset]])

train_loader = torch_data.DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=4
)
val_loader = torch_data.DataLoader(
    val_dataset, batch_size=8, shuffle=True, num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(1, 2).to(device)

criterion = losses.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
baseline_trainer = trainer.Trainer(
    model, optimizer, criterion, train_loader, val_dataloader=val_loader, device=device
)
baseline_trainer.train(10)
