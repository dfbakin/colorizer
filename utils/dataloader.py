import glob
import os
import urllib.request
import zipfile

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


class ColorizationDataset(Dataset):
    def __init__(
        self, images_dir, resize=(256, 256), classes_folders=False
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
        return img[:, :, 0], img[:, :, 1:]  # input: L*, output: a* and b* channels


if __name__ == "__main__":
    coco_dataset = "val2017"
    coco_url = "http://images.cocodataset.org/zips/" + coco_dataset + ".zip"
    coco_path = "./coco_data"
    coco_extract_path = os.path.join(coco_path, coco_dataset)

    if not os.path.exists(coco_path):
        download_and_extract(coco_url, coco_path)
    else:
        print("COCO dataset already extracted.")

    dataset = ColorizationDataset(coco_extract_path)

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Total images: {len(dataset)}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    for gray, color in train_loader:
        print(f"Gray image shape: {gray.shape}")
        print(f"Color image shape: {color.shape}")
        break
