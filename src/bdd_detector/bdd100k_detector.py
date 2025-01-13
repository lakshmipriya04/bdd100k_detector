# Contains code for BDD100K object detection pipeline for creating yolo labels, training model
import os
from pathlib import Path
from typing import Any, List

import cv2
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from bdd_dataset import BDD100KDataset
from bdd_detector import TrainingArguments
from bdd_trainer import Trainer
from bdd_utils import file_utils, data_utils


def collate_fn(batch):
    images, bboxes, labels = zip(*batch)
    return torch.stack(images), list(bboxes), list(labels)


class ObjectDetection:
    def __init__(self, args: TrainingArguments):
        """
        Base EDA class for initializing configuration.
        Args:
            config: A path to a YAML file or a dictionary with configuration.
        """
        self.args = args
        config = args.data_config_path
        if isinstance(config, str):
            self.config = file_utils.load_config(config)
        else:
            raise ValueError("Config must be a YAML file path or a dictionary.")
        print(self.config)
        self.train_labels = self.config.get("train_label")
        self.val_labels = self.config.get("val_label")
        self.train_images = self.config.get("train_images")
        self.val_images = self.config.get("val_images")
        self.train_csv = self.config.get("train_csv")
        self.val_csv = self.config.get("val_csv")
        self.device = "0" if torch.cuda.is_available() else "cpu"
        data_utils.prepare_file_path(self.train_csv)
        data_utils.prepare_file_path(self.val_csv)
        self._validate_config()

    def _validate_config(self):
        """Validate the required configuration keys."""
        required_keys = ["train_label", "val_label", "train_images", "val_images"]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config key: {key}")

    def load_labels(self, file_path: str) -> Any:
        """Load a JSON label file."""
        return file_utils.read_json(file_path)

    def __repr__(self):
        return f"BaseEDA(config={self.config})"


class BDD100KDetection(ObjectDetection):
    def __init__(self, args: TrainingArguments, eda, trainer: Trainer):
        super().__init__(args)
        self.output_path = args.images_save_path
        self.args = args
        self.trainer = trainer
        os.makedirs(self.output_path, exist_ok=True)
        # Initialize dataframe attributes to None
        self.train_df = None
        self.val_df = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.df_train_val = None
        self.train_dataset = None
        self.val_dataset = None
        self._create_dataframe()
        self.labels2idx()
        self.idx2labels()
        self.add_bbox_attributes()

    def _convert2dataframe(self, data):
        # Initialize lists to store data
        data_flattened = []

        # Iterate over the image and labels
        for item in data:
            image_info = {
                'image_id': item['name'],
                'weather': item['attributes'].get('weather', None),
                'timeofday': item['attributes'].get('timeofday', None),
                'scene': item['attributes'].get('scene', None),
                'timestamp': item['timestamp'],
            }
            if 'labels' not in item:
                obj_info = {
                    'image_id': item['name'],
                    'category': None,
                    'traffic_light_color': None,
                    'occluded': None,
                    'truncated': None,
                    'bbox_x1': None,
                    'bbox_y1': None,
                    'bbox_x2': None,
                    'bbox_y2': None,
                }
                data_flattened.append({**image_info, **obj_info})
                continue
            for obj in item['labels']:
                obj_info = {
                    'image_id': item['name'],
                    'category': obj['category'],
                    'traffic_light_color': obj['attributes'].get('trafficLightColor', None),
                    'occluded': obj['attributes'].get('occluded', None),
                    'truncated': obj['attributes'].get('truncated', None),
                    'bbox_x1': obj['box2d']['x1'],
                    'bbox_y1': obj['box2d']['y1'],
                    'bbox_x2': obj['box2d']['x2'],
                    'bbox_y2': obj['box2d']['y2']
                }
                # Combine image info and object info
                data_flattened.append({**image_info, **obj_info})
        return pd.DataFrame(data_flattened)

    def _create_dataframe(self):
        if os.path.exists(self.train_csv) and os.path.exists(self.val_csv):
            self.train_df = pd.read_csv(self.train_csv)
            self.val_df = pd.read_csv(self.val_csv)
            self.df_train_val = pd.concat([self.train_df, self.val_df])
        else:
            train_labels = file_utils.read_json(self.train_labels)
            val_labels = file_utils.read_json(self.val_labels)

            self.train_df = self._convert2dataframe(train_labels)
            self.val_df = self._convert2dataframe(val_labels)

            self.train_df['split'] = 'train'
            self.train_df['image_path'] = self.train_images
            # self.train_df = data_utils.add_image_dimensions(self.train_df,self.train_images)
            self.train_df['img_width'] = 1280
            self.train_df['img_height'] = 720

            self.val_df['split'] = 'val'
            self.val_df['image_path'] = self.val_images
            # self.val_df = data_utils.add_image_dimensions(self.val_df,self.val_images)
            self.val_df['img_width'] = 1280
            self.val_df['img_height'] = 720

            self.df_train_val = pd.concat([self.train_df, self.val_df])
            self.train_df.to_csv(self.train_csv, index=False)
            self.val_df.to_csv(self.val_csv, index=False)

    def labels2idx(self):
        """
        Create a dictionary mapping labels (categories) to indices based on unique categories in train_df.
        Labels are sorted alphabetically.
        """
        if self.train_df is None:
            raise ValueError("Train DataFrame is not initialized.")

        unique_labels = sorted(self.train_df['category'].dropna().unique())  # Drop NaN and sort alphabetically
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return self.label_to_idx

    def idx2labels(self):
        """
        Create a dictionary mapping indices back to label (categories).
        This is the inverse of labels2idx.
        """
        if not self.label_to_idx:
            self.labels2idx()  # Ensure the label-to-index mapping exists

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        return self.idx_to_label

    def add_bbox_attributes(self):
        """
        Adds columns for center_x, center_y, width, height, and their normalized values
        (center_x_norm, center_y_norm, width_norm, height_norm) to the DataFrame.
        """
        if self.df_train_val is None:
            raise ValueError("Dataframe is not initialized.")

        # Compute center_x, center_y, width, and height
        self.df_train_val['center_x'] = (self.df_train_val['bbox_x1'] + self.df_train_val['bbox_x2']) / 2
        self.df_train_val['center_y'] = (self.df_train_val['bbox_y1'] + self.df_train_val['bbox_y2']) / 2
        self.df_train_val['width'] = self.df_train_val['bbox_x2'] - self.df_train_val['bbox_x1']
        self.df_train_val['height'] = self.df_train_val['bbox_y2'] - self.df_train_val['bbox_y1']

        # Add normalized values
        self.df_train_val['center_x_norm'] = self.df_train_val['center_x'] / self.df_train_val['img_width']
        self.df_train_val['center_y_norm'] = self.df_train_val['center_y'] / self.df_train_val['img_height']
        self.df_train_val['width_norm'] = self.df_train_val['width'] / self.df_train_val['img_width']
        self.df_train_val['height_norm'] = self.df_train_val['height'] / self.df_train_val['img_height']
        self.df_train_val['category_idx'] = self.df_train_val['category'].map(self.label_to_idx)
        self.train_df = self.df_train_val[self.df_train_val['split'] == 'train']
        self.val_df = self.df_train_val[self.df_train_val['split'] == 'val']

    def visualize_samples(self, split: str, indices: List[int], display_image: bool = False, save_path: str = None):
        """
        Visualize sample images from the specified split (train/val).

        Args:
            split (str): "train" or "val" to select the data split.
            indices (list): List of indices to visualize.
            display_image (bool): Whether to display the image.
            save_path (str): Path to save the visualization.
        """
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:
            raise ValueError("Split must be 'train' or 'val'.")

        # Call the visualize method from the BDD100KDataset instance
        dataset.visualize(indices, display_image, save_path)

    def create_dataloader(self, split: str, batch_size: int, shuffle: bool = True, transform=None):
        """
        Create a PyTorch DataLoader for a given split.

        Args:
            split (str): "train" or "val".
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.
            transform: Transformations to apply to images.

        Returns:
            DataLoader: PyTorch DataLoader.
        """
        if split == "train":
            dataframe = self.train_df
            dataset = self.train_dataset  # Reuse the dataset if already created
            if not dataset:  # Check if dataset is already created
                self.train_dataset = BDD100KDataset(dataframe, transform=transform)
                dataset = self.train_dataset
        elif split == "val":
            dataframe = self.val_df
            dataset = self.val_dataset  # Reuse the dataset if already created
            if not dataset:  # Check if dataset is already created
                self.val_dataset = BDD100KDataset(dataframe, transform=transform)
                dataset = self.val_dataset
        else:
            raise ValueError("Split must be 'train' or 'val'.")
        # breakpoint()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    def create_yolo_config(self, yaml_path: str, force=False):
        """
        Create a yaml file for BDD100K in a format similar to COCO YAML.

        Args:
        yaml_path (str): The path where the YAML file will be saved.
        """
        if os.path.exists(yaml_path) and not force:
            return True
        # Ensure label_to_idx is populated before proceeding
        if not self.label_to_idx:
            self.labels2idx()

        if not self.idx_to_label:
            self.idx2labels()

            # Get the relative paths for train and val sets
        train_images_path = self.train_images
        val_images_path = self.val_images

        # Check if paths are valid (this is optional, but good for robustness)
        if not os.path.exists(train_images_path) or not os.path.exists(val_images_path):
            raise FileNotFoundError("Train or Validation images directory not found.")

        train_path = Path(train_images_path)
        val_path = Path(val_images_path)
        # Get the components of the path
        train_components = list(train_path.parts)
        val_components = list(val_path.parts)
        index_images = train_components.index('images')
        # Get the path before 'images'
        root_path = Path(*train_components[:index_images])
        relative_train_path = str(Path(*train_components[index_images:]))
        relative_val_path = str(Path(*val_components[index_images:]))
        # Construct the YAML data
        yaml_data = {
            'path': str(root_path),  # Dataset root path
            'train': relative_train_path,  # Path to training images
            'val': relative_val_path,  # Path to validation images
            'test': None,  # Optional, as there's no test set defined
            'names': self.idx_to_label  # Class name mapping (from idx_to_label)
        }

        # Write the YAML data to the specified file
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)

        print(f"YAML file has been created at: {yaml_path}")

    def create_labels(self, force: bool = False):
        """
        Create label files for each image in the dataset.
        This method generates label files for both the training and validation sets
        based on the information in the dataframe and the paths to images.

        Args:
            force (bool): If True, overwrite existing label files. Otherwise, skip existing files.
        """
        if self.df_train_val is None:
            raise ValueError("Dataframe is not initialized.")

        # Initialize a set of existing label files
        existing_labels = set()
        for split in ['train', 'val']:
            label_dir = os.path.join(self.train_images if split == 'train' else self.val_images).replace('images',
                                                                                                         'labels')
            for root, _, files in os.walk(label_dir):
                for file in files:
                    existing_labels.add(os.path.join(root, file))

        # Group the DataFrame by 'image_id' to process annotations per image
        grouped = self.df_train_val.groupby('image_id')

        for image_id, group in tqdm(grouped, total=len(grouped), desc="Creating labels"):
            # Get the split from the first row (all rows in a group share the same split)
            split = group.iloc[0]['split']

            # Determine the label directory and file path
            image_dir = self.train_images if split == 'train' else self.val_images
            label_dir = image_dir.replace('images', 'labels')
            label_name = image_id.replace('.jpg', '.txt')
            label_file_path = os.path.join(label_dir, label_name)

            # Skip if the label file exists and force is False
            if label_file_path in existing_labels and not force:
                continue

            # Ensure the label directory exists
            os.makedirs(os.path.dirname(label_file_path), exist_ok=True)

            # Collect all bounding box annotations for this image
            label_data = []
            for _, row in group.iterrows():
                if row['category'] is not None:  # Only add labels with a valid category
                    category_idx = self.label_to_idx.get(row['category'])
                    if category_idx is not None:
                        # Normalize the bounding box coordinates
                        center_x = row['center_x_norm']
                        center_y = row['center_y_norm']
                        width = row['width_norm']
                        height = row['height_norm']

                        # Format the YOLO annotation line
                        label_data.append(f"{category_idx} {center_x} {center_y} {width} {height}")
            # Write all annotations for this image to the label file
            if label_data:
                with open(label_file_path, 'w') as label_file:
                    label_file.write("\n".join(label_data))

    def check_yolo_labels(self, num_samples: int, output_path: str):
        """
        Visualize bounding boxes from the label text files and save images with boxes drawn.
        
        Args:
            num_samples (int): The number of unique image_id samples to visualize.
            output_path (str): The directory to save the visualized images with bounding boxes.
        """
        if self.df_train_val is None:
            raise ValueError("Dataframe is not initialized.")
        if output_path:
            # Make sure the output directory exists
            os.makedirs(output_path, exist_ok=True)

        # Group by image_id and randomly sample num_samples unique image_id
        sampled_image_ids = self.df_train_val['image_id'].drop_duplicates().sample(num_samples)
        sampled_df = self.df_train_val[self.df_train_val['image_id'].isin(sampled_image_ids)]

        for image_id in sampled_image_ids:
            # Filter rows corresponding to the current image_id
            image_data = sampled_df[sampled_df['image_id'] == image_id]

            # Determine the split (assuming it's the same for all rows with the same image_id)
            split = image_data.iloc[0]['split']

            # Determine the image path
            image_dir = self.train_images if split == 'train' else self.val_images
            image_path = os.path.join(image_dir, image_id)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {image_path} not found or could not be loaded.")
                continue

            # Derive the label file path
            label_dir = image_dir.replace('images', 'labels')
            label_file_path = os.path.join(label_dir, image_id.replace('.jpg', '.txt'))

            # Check if label file exists
            if not os.path.exists(label_file_path):
                print(f"Label file {label_file_path} not found.")
                continue

            # Read the label data from the text file
            with open(label_file_path, 'r') as label_file:
                label_data = label_file.readlines()

            # Draw bounding boxes for each object in the label file
            for label in label_data:
                parts = label.strip().split()
                category_idx = int(parts[0])
                center_x, center_y, width, height = map(float, parts[1:])

                # Convert to pixel values
                img_width = image_data.iloc[0]['img_width']
                img_height = image_data.iloc[0]['img_height']
                x1 = int((center_x - width / 2) * img_width)
                y1 = int((center_y - height / 2) * img_height)
                x2 = int((center_x + width / 2) * img_width)
                y2 = int((center_y + height / 2) * img_height)

                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

                category_name = self.idx_to_label.get(category_idx, "Unknown")
                image = cv2.putText(image, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            if output_path:
                # Save the image with bounding boxes
                output_image_path = os.path.join(output_path, f"{image_id.replace('.jpg', '')}_with_boxes.jpg")
                cv2.imwrite(output_image_path, image)

                print(f"Saved visualized image to {output_image_path}")

    def train(self):
        return self.trainer.train()

    def validate(self):
        self.trainer.evaluate()
