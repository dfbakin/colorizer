import glob
import os
import urllib.request
import zipfile

import albumentations
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np


def download_and_extract(url, path):
    zip_path = os.path.join(path, "dataset.zip")
    os.makedirs(path, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        with tqdm(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading"
        ) as t:

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    t.total = total_size
                t.update(block_size)

            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        print("Download completed.")
    else:
        print("Zip file already exists. Skipping download.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path)
    print("Extraction completed.")

    os.remove(zip_path)

def colorfulness_metric(image):
    """
    Compute a measure of colorfulness based on the standard deviation 
    and mean of opponent color channels.
    """
    (B, G, R) = cv2.split(image.astype("float"))
    
    # Compute color differences
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    
    # Compute mean and standard deviation
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    
    # Final colorfulness metric
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    
    return colorfulness

def is_colorful(image, threshold=1):
    """
    Determines if an image is colorful based on a set threshold.
    """
    return colorfulness_metric(image) > threshold


class ColorizationDataset(Dataset):
    ab_norm_factor = 256.0
    l_norm_factor = 256.0

    def __init__(
        self, images_dir, resize=(256, 256), classes_folders=False, filter_colorless=True
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

        # resize_trasform_list = [A.Resize(width=resize[0], height=resize[1])]
        #           if self.new_size is not None else []

        # if transform_color is None:
        #     self.transform_color = A.Compose(
        #         resize_trasform_list +
        #         [
        #             A.ToTensorV2(),
        #         ]
        #     )
        # else:
        #     self.transform_color = transform_color

        # if transform_gray is None:
        #     self.transform_gray = A.Compose(
        #         resize_trasform_list +
        #         [
        #             A.ToGray(num_output_channels=1, p=1.0),
        #             A.ToTensorV2(),
        #         ]
        #     )
        # else:
        #     self.transform_gray = transform_gray

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.images_paths[idx])
        if self.new_size is not None:
            img = cv2.resize(img, self.new_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        tensor_img = self.to_tensor_albumentation.apply(img).to(torch.float32)
        # print(tensor_img[0:1, :, :].shape, tensor_img[1:, :, :].dtype)
        return tensor_img[0:1, :, :] / self.l_norm_factor, tensor_img[
            1:, :, :
        ] / self.ab_norm_factor  # input: L*, output: a* and b* channels
    
    def get_color_metric(self, idx):
        img = cv2.imread(self.images_paths[idx])
        return colorfulness_metric(img)

    @staticmethod
    def torch_L_ab_to_cvimage(L_input, ab_input, denorm=True):
        print(L_input.shape, ab_input.shape)
        # expected batches shape: [B, 1, H, W] and [B, 2, H, W]

        cielab_img = torch.cat(
            [
                L_input * (ColorizationDataset.l_norm_factor if denorm else 1.0),
                ab_input * (ColorizationDataset.ab_norm_factor if denorm else 1.0),
            ],
            dim=1,
        ).numpy()
        cielab_img = cielab_img.transpose(0, 2, 3, 1)
        # reference "RGB ↔ CIE L*u*v*" from https://docs.opencv.org/4.8.0/de/d25/imgproc_color_conversions.html
        cielab_img = np.clip(cielab_img, 0, 255).astype(np.uint8)
        result = np.zeros_like(cielab_img, dtype=np.uint8)
        print(result.shape)
        print(cielab_img.shape)
        for i in range(cielab_img.shape[0]):
            result[i, :, :, :] = cv2.cvtColor(
                cielab_img[i, :, :, :], cv2.COLOR_LAB2RGB
            )
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

        return tensor_img[:, 0, :, :] / ColorizationDataset.l_norm_factor, tensor_img[
            :, 1:, :, :
        ] / ColorizationDataset.ab_norm_factor


if __name__ == "__main__":
    print("Use DVC to download the dataset")
    raise SystemExit
    # coco_dataset = "val2017"
    # coco_url = "http://images.cocodataset.org/zips/" + coco_dataset + ".zip"
    # coco_path = "./coco_data"
    # coco_extract_path = os.path.join(coco_path, coco_dataset)

    # if not os.path.exists(coco_path):
    #     download_and_extract(coco_url, coco_path)
    # else:
    #     print("COCO dataset already extracted.")

    # dataset = ColorizationDataset(coco_extract_path)

    # torch.manual_seed(42)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, test_size]
    # )

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # print(f"Total images: {len(dataset)}")
    # print(f"Train images: {len(train_dataset)}")
    # print(f"Test images: {len(test_dataset)}")

    # for gray, color in train_loader:
    #     print(f"Gray image shape: {gray.shape}")
    #     print(f"Color image shape: {color.shape}")
    #     break
