import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
import plotly.express as px
from torchvision.io import decode_image
from torchvision.utils import draw_bounding_boxes

from bdd_utils import file_utils

plt.rcParams["savefig.bbox"] = "tight"


def display_bounding_boxes(
    df: pd.DataFrame, image_wise: bool, num_samples: int = 20
) -> None:
    """
    Display bounding boxes on images based on the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing columns `image_id`, `image_path`, 
                           `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`.
        image_wise (bool): If True, group by `image_id` and sample groups up to `num_samples`.
                           If False, randomly sample rows from the DataFrame.
        num_samples (int): The number of samples to display.

    Returns:
        None
    """
    if image_wise:
        # Group by image_id and sample up to `num_samples` groups
        grouped = df.groupby("image_id")
        sampled_groups = random.sample(list(grouped), min(num_samples, len(grouped)))

        for image_id, group in sampled_groups:
            image_path = os.path.join(group["image_path"].iloc[0],image_id)  # Assume all rows in the group have the same image_path
            plot_image_with_bboxes(image_path, group)
    else:
        # Randomly sample rows without grouping
        sampled_rows = df.sample(n=min(num_samples, len(df)))
        for _, row in sampled_rows.iterrows():
            image_path = os.path.join(row["image_path"],row['image_id'])
            plot_image_with_bboxes(image_path, pd.DataFrame([row]))

def plot_image_with_bboxes(image_path: str, group: pd.DataFrame, save_path=None) -> None:
    """
    Helper function to plot bounding boxes on a single image.

    Args:
        image_path (str): Path to the image file.
        group (pd.DataFrame): DataFrame containing bounding box information for the image.

    Returns:
        None
    """
    try:
        # Read the image using cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to read image at {image_path}")
            return

        # Convert BGR to RGB (matplotlib expects RGB format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes
        for _, row in group.iterrows():
            x1, y1, x2, y2 = row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0), thickness=2)

        # Display the image with bounding boxes
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image: {group['image_id'].iloc[0]}")
        plt.show()
        if save_path:
            if os.path.isdir(save_path):  # If save_path is a directory
                # Create the directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)
                # Construct the file path
                filename = os.path.basename(image_path)
                save_file_path = os.path.join(save_path, filename)
            else:
                save_file_path = save_path  # Use the file name directly

            # Convert the image back to BGR before saving with OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_file_path, img_bgr)
            print(f"Image saved at {save_file_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


def draw_gt_pred_boxes(gt_df: pd.DataFrame, pred_df: pd.DataFrame, image_id: str, image_dir: str, output_path: str = None):
    """
    Draw ground truth and predicted bounding boxes on an image. Optionally save the image or display it.

    Args:
        gt_df (pd.DataFrame): DataFrame containing ground truth annotations with columns:
                              ['image_id', 'category_id', 'x', 'y', 'width', 'height', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'].
        pred_df (pd.DataFrame): DataFrame containing predicted annotations with similar columns as `gt_df`.
        image_id (str): The ID of the image to draw bounding boxes for.
        image_dir (str): Directory containing the images.
        output_path (str, optional): Path to save the image with bounding boxes. If None, the image will be displayed.

    """
    # Get the ground truth and prediction rows for the given image_id
    gt_boxes = gt_df[gt_df['image_id'] == image_id]
    pred_boxes = pred_df[pred_df['image_id'] == image_id]
    
    # Construct the path to the image
    image_path = os.path.join(image_dir, f"{image_id}")  # Adjust the extension if necessary
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found or could not be loaded.")
    
    # Draw ground truth boxes in green
    for _, row in gt_boxes.iterrows():
        x1, y1, x2, y2 = int(row['bbox_x1']), int(row['bbox_y1']), int(row['bbox_x2']), int(row['bbox_y2'])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Draw predicted boxes in red
    for _, row in pred_boxes.iterrows():
        x1, y1, x2, y2 = int(row['bbox_x1']), int(row['bbox_y1']), int(row['bbox_x2']), int(row['bbox_y2'])
        # score = row['score']
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

    # If output_path is not provided, display the image using OpenCV
    if output_path is None:
        # Convert BGR to RGB for display with matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"Image: {image_id}")
        plt.show()
    else:
        # Save the output image
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{image_id}")
        cv2.imwrite(output_file, image)
        print(f"Saved image with bounding boxes to {output_file}")


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def display_image_with_boxes(image_np, boxes, colors="red"):
    """
    Displays an image with bounding boxes drawn on it.

    Args:
        image_np (numpy.ndarray): The image in (H, W, C) format.
        boxes (torch.Tensor): Bounding boxes of shape (N, 4), where N is the number of predictions and 4 corresponds to (x1, y1, x2, y2).
        colors (str or list): Color for the bounding boxes.
    """
    # Convert the NumPy array to a PyTorch tensor and permute to (C, H, W)
    image_tensor = torch.from_numpy(image_np).float()

    # Normalize the image to the range [0, 1] if it's in the range [0, 255]
    if image_tensor.max() > 1:  # Check if image tensor has values > 1 (assumed 0-255 range)
        image_tensor /= 255.0

    # If the image was originally in (H, W, C), PyTorch expects (C, H, W), so permute
    image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)

    # Draw bounding boxes (ensure the box coordinates are in xyxy format)
    drawn_boxes = draw_bounding_boxes(image_tensor, boxes, colors=colors)

    # Convert back to (H, W, C) for displaying
    drawn_boxes = drawn_boxes.permute(1, 2, 0)  # Convert back to (H, W, C)

    # If image is normalized to [0, 1], multiply by 255 to restore [0, 255] range
    if drawn_boxes.max() <= 1:
        drawn_boxes = (drawn_boxes * 255).byte()
    else:
        drawn_boxes = drawn_boxes.byte()

    # Show the result using matplotlib
    plt.imshow(drawn_boxes)
    plt.axis('off')  # Turn off axis
    plt.show()

def display_bounding_box_for_sample(df, conditions: dict, num_samples: int = 8, sample_index: int = 4):
    """
    Displays bounding boxes for multiple samples from the filtered dataset.
    
    Args:
    - df (DataFrame): The dataframe containing the image data.
    - conditions (dict): A dictionary containing the conditions to filter the dataframe. 
                          The keys are column names and values are the corresponding values to filter by.
    - num_samples (int): Number of samples to randomly pick from the filtered data.
    - sample_index (int): The index of the sample to display from the randomly chosen set.

    Returns:
    - None
    """
    # Apply all conditions from the dictionary to filter the dataframe
    filtered_df = df
    for column, value in conditions.items():
        filtered_df = filtered_df[filtered_df[column] == value]
    
    # Randomly select a sample
    random_samples = filtered_df.sample(n=num_samples)

    # Plotting setup
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    axes = axes.flatten()

    # Iterate over the samples and display bounding boxes
    for i, (idx, row) in enumerate(random_samples.iterrows()):
        # Retrieve the image path and decode the image
        image_path = file_utils.find_image_and_read(row['image_path'], row['image_id'])
        img_int = decode_image(image_path)
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = row[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
        box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
        
        # Draw bounding box on the image
        result = draw_bounding_boxes(img_int, box, colors=['red'], width=6)
        
        # Plot the image with annotation
        axes[i].imshow(result.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for matplotlib
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i+1} Annotation")

    plt.tight_layout()
    plt.show()


def plot_distribution(
            df: pd.DataFrame,
            group_by_column: str,
            title: str,
            x_label: str,
            output_filename: str=None,
            save_path=None
    ):
        """
        Plot the distribution of a given column as percentages in both train and validation datasets.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            group_by_column (str): The column to group by (e.g., 'weather', 'scene').
            title (str): The title of the plot.
            output_filename (str): The name of the file to save the plot.
            x_label (str): The label for the x-axis.
        """
        # Group by the given column and 'split', and count unique 'image_id'
        counts = df.groupby([group_by_column, 'split'])['image_id'].nunique().reset_index(name='count')

        # Calculate percentages for each split
        total_counts = counts.groupby('split')['count'].transform('sum')  # Total count per split
        counts['percentage'] = (counts['count'] / total_counts) * 100  # Convert to percentage

        # Add a column for displaying both percentage and count
        counts['text'] = counts.apply(
            lambda row: f"{row['count']} ({row['percentage']:.2f}%)", axis=1
        )

        # Create a grouped bar plot
        fig = px.bar(
            counts,
            x=group_by_column,
            y='percentage',
            color='split',
            barmode='group',
            title=title,
            labels={
                'percentage': 'Percentage of Images (%)',
                group_by_column: x_label,
                'split': 'Dataset',
            },
            text='text',  # Display both count and percentage on the bars
        )

        # Update layout for better visualization
        fig.update_traces(texttemplate='%{text}', textposition='outside')  # Format text to show count and percentage
        fig.update_layout(
            yaxis=dict(title='Percentage of Images (%)', tickformat=".2f"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        if save_path:
            # Save the plot to the output directory
            png_path = f"{save_path}/{output_filename}.png"
            fig.write_image(png_path, format="png", width=800, height=600)

        # Show the plot
        fig.show()
