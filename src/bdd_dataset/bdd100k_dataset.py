# Contains code for creating pytorch dataset of BDD100K data
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes, make_grid


class BDD100KDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get row corresponding to index
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(row['image_path'], row['image_id'])
        image = Image.open(image_path).convert("RGB")

        # Bounding box and label
        bbox = torch.tensor([
            row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
        ], dtype=torch.float32)
        label = torch.tensor(row['category_idx'], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return to_tensor(image), bbox, label

    def visualize(self, indices: List[int], display_image: bool = False, save_path: str = None):
        """
        Visualize images with annotations using normalized coordinates.

        Args:
            indices (list): List of indices to visualize.
        """
        images_with_boxes = []
        for idx in indices:
            row = self.dataframe.iloc[idx]
            image_path = os.path.join(row['image_path'], row['image_id'])
            image = Image.open(image_path).convert("RGB")
            image_tensor = to_tensor(image)

            # Convert normalized coordinates to pixel coordinates
            img_width, img_height = row['img_width'], row['img_height']
            center_x = row['center_x_norm'] * img_width
            center_y = row['center_y_norm'] * img_height
            width = row['width_norm'] * img_width
            height = row['height_norm'] * img_height
            x1 = center_x - (width / 2)
            y1 = center_y - (height / 2)
            x2 = center_x + (width / 2)
            y2 = center_y + (height / 2)

            # Draw bounding box
            box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            image_with_box = draw_bounding_boxes(
                (image_tensor * 255).to(torch.uint8),
                box,
                colors=["red"],
                width=3
            )
            images_with_boxes.append(image_with_box)

        # Create a grid of images
        grid = make_grid(images_with_boxes, nrow=4)
        if save_path:
            # Save the grid to the specified path
            grid_image = grid.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
            plt.imsave(save_path, grid_image.astype("uint8"))
            print(f"Visualization grid saved to {save_path}")

        if display_image:
            # Display the grid of annotated images
            plt.figure(figsize=(15, 8))
            plt.axis("off")
            plt.title("Annotated Images")
            plt.imshow(grid.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
            plt.show()
