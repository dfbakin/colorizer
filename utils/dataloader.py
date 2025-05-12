import glob
import os

import albumentations
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def colorfulness_metric(image):
    """
    Compute a measure of colorfulness based on the standard deviation
    and mean of opponent color channels.
    """
    (B, G, R) = cv2.split(image.astype("float"))

    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)

    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(
        mean_rg**2 + mean_yb**2
    )

    return colorfulness


def is_colorful(image, threshold=1):
    return colorfulness_metric(image) > threshold


class ColorizationDataset(Dataset):
    ab_reduce = 40
    ab_norm_factor = 256.0
    l_norm_factor = 180.0

    def __init__(
        self,
        images_dir,
        resize=(256, 256),
        classes_folders=False,
        filter_colorless=True,
    ):  # transform_color=None, transform_gray=None,
        if not classes_folders:
            self.images_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
        else:
            self.images_paths = []
            for class_folder in os.listdir(images_dir):
                class_folder_path = os.path.join(images_dir, class_folder)
                self.images_paths += sorted(
                    glob.glob(os.path.join(class_folder_path, "*"))
                )

        self.new_size = resize
        # TODO create a custom albumentation transform to wrap OpenCV cvtColor
        self.to_tensor_albumentation = albumentations.ToTensorV2()

        if not filter_colorless:
            return

        self.colorful_images = []
        for idx in tqdm(range(len(self.images_paths))):
            path = self.images_paths[idx]
            img = cv2.imread(path)  # Read image
            if img is not None and is_colorful(img):
                self.colorful_images.append(path)
            if img is not None:
                del img
        self.images_paths = self.colorful_images

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.images_paths[idx])
        if self.new_size is not None:
            img = cv2.resize(img, self.new_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        tensor_img = self.to_tensor_albumentation.apply(img).to(torch.float32)
        return ColorizationDataset.normalize(tensor_img)

    def get_color_metric(self, idx):
        img = cv2.imread(self.images_paths[idx])
        return colorfulness_metric(img)

    @staticmethod
    def torch_L_ab_to_cvimage(L_input, ab_input, denorm=True):
        # expected batches shape: [B, 1, H, W] and [B, 2, H, W]

        cielab_img = torch.cat(
            [
                L_input * (ColorizationDataset.l_norm_factor if denorm else 1.0),
                ab_input * (ColorizationDataset.ab_norm_factor if denorm else 1.0)
                + ColorizationDataset.ab_reduce * int(denorm),
            ],
            dim=1,
        ).numpy()
        cielab_img = cielab_img.transpose(0, 2, 3, 1)
        # reference "RGB ↔ CIE L*u*v*" from https://docs.opencv.org/4.8.0/de/d25/imgproc_color_conversions.html
        cielab_img = np.clip(cielab_img, 0, 255).astype(np.uint8)
        result = np.zeros_like(cielab_img, dtype=np.uint8)
        for i in range(cielab_img.shape[0]):
            result[i, :, :, :] = cv2.cvtColor(cielab_img[i, :, :, :], cv2.COLOR_LAB2RGB)
        return result

    @staticmethod
    def cvimage_to_torch_L_ab(cv_images_batch):
        # expected batches shape RGB images: [B, H, W, 3]
        for i in range(cv_images_batch.shape[0]):
            cv_images_batch[i, :, :, :] = cv2.cvtColor(
                cv_images_batch[i, :, :, :], cv2.COLOR_LAB2RGB
            )

        tensor_img = torch.tensor(
            cv_images_batch.transpose(0, 3, 1, 2), dtype=torch.float32
        )

        return ColorizationDataset.normalize(tensor_img)

    @staticmethod
    def normalize(Lab_tensor):
        CD = ColorizationDataset
        if len(Lab_tensor.shape) == 4:
            L = Lab_tensor[:, 0, :, :] / CD.l_norm_factor
            ab = (Lab_tensor[:, 1:, :, :] - CD.ab_reduce) / CD.ab_norm_factor
        elif len(Lab_tensor.shape) == 3:
            L = Lab_tensor[0:1, :, :] / CD.l_norm_factor
            ab = (Lab_tensor[1:, :, :] - CD.ab_reduce) / CD.ab_norm_factor
        else:
            raise ValueError("Invalid shape for Lab tensor")

        return L, ab


if __name__ == "__main__":
    print("Use DVC to download the dataset")
    raise SystemExit
