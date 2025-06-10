import sys
import torch
import numpy as np
import cv2

from io import BytesIO
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from models import UNetGANWrap
from utils.dataloader import ColorizationDataset
from fastai.vision.models import resnet18
from fastai.vision.models.unet import DynamicUnet
import torch.nn as nn

def build_res_unet(n_input=1, n_output=2, size=256, freeze_encoder=True):
    encoder = resnet18(weights="IMAGENET1K_V1")
    layers = [nn.Conv2d(n_input, 3, kernel_size=1)] + list(encoder.children())[:-2]
    encoder = nn.Sequential(*layers)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
    return DynamicUnet(encoder, n_output, (size, size), norm_type=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_res_unet(n_input=1, n_output=2, size=256, freeze_encoder=True)
model.load_state_dict(torch.load("checkpoints/baseline/net_G.pth", map_location=device))
model.eval()
model.to(device)

def preprocess_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    tensor_lab = torch.tensor(img_lab.transpose(2, 0, 1), dtype=torch.float32)
    L, _ = ColorizationDataset.normalize(tensor_lab)
    return L.unsqueeze(0), (w, h)

def postprocess_image(L: torch.Tensor, ab: torch.Tensor, orig_size):
    cv_img = ColorizationDataset.torch_L_ab_to_cvimage(L, ab)[0]
    cv_img = cv2.resize(cv_img, orig_size)
    _, buf = cv2.imencode(".jpg", cv_img)
    return buf.tobytes()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Send me a grayscale image and I’ll colorize it for you.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    image_bytes = bio.getvalue()

    L, orig_size = preprocess_image(image_bytes)

    with torch.no_grad():
        L_device = L.to(device)
        
        fake = model(L_device)
        print(L_device.shape, fake.shape)

    colored = postprocess_image(L, fake.cpu(), orig_size)

    await update.message.reply_photo(photo=colored)

def main():
    token = sys.argv[1]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_image))

    app.run_polling()

if __name__ == "__main__":
    main()
