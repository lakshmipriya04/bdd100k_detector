# Contains common functions for data related operations
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image


# Helper function to get image dimensions
def get_image_dimensions(image_id: str, images_path: str) -> tuple:
    image_path = Path(images_path) / image_id
    if image_path.exists():
        try:
            with Image.open(image_path) as img:
                return img.width, img.height
        except Exception as e:
            # Log any exceptions (corrupt image, etc.)
            print(f"Error opening image {image_id}: {e}")
            return None, None
    else:
        # If image doesn't exist, return None
        return None, None


# Function to apply the helper function in parallel
def add_image_dimensions(df: pd.DataFrame, images_path: str) -> pd.DataFrame:
    # Parallelize the operation with ThreadPoolExecutor to avoid I/O blocking
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda image_id: get_image_dimensions(image_id, images_path), df['image_id']))

    # Create new columns for width and height
    df[['width', 'height']] = pd.DataFrame(results, columns=['width', 'height'])

    return df


def prepare_file_path(file_path: str):
    """
    Ensures the parent directory of a given file path exists.
    If not, it creates the directory.

    Args:
        file_path (str): The file path whose parent directory needs to be ensured.
    """
    path = Path(file_path)
    if not path.parent.exists():
        print(f'Path does not exists {path.parent}')
        path.parent.mkdir(parents=True, exist_ok=True)

def get_min_max_rows(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """
    Get the rows corresponding to the minimum and maximum bounding box areas for each category.

    Args:
        df (pd.DataFrame): The input DataFrame containing bounding box data.
        category_col (str): The column name for the category.

    Returns:
        pd.DataFrame: A DataFrame containing rows for minimum and maximum area for each category.
    """
    # Compute the bounding box area
    df['bbox_area'] = (df['bbox_x2'] - df['bbox_x1']) * (df['bbox_y2'] - df['bbox_y1'])

    # Find the indices of rows with min and max area for each category
    min_indices = df.groupby(category_col)['bbox_area'].idxmin()
    max_indices = df.groupby(category_col)['bbox_area'].idxmax()

    # Combine the rows into a single DataFrame
    min_rows = df.loc[min_indices]
    max_rows = df.loc[max_indices]

    # Add a new column to indicate whether it's min or max
    min_rows['area_type'] = 'min'
    max_rows['area_type'] = 'max'

    # Concatenate the two DataFrames
    result_df = pd.concat([min_rows, max_rows]).reset_index(drop=True)

    return result_df


def create_predictions_dataframe(predictions_folder: str, img_size: Tuple[int, int] = (1280, 720)) -> pd.DataFrame:
    """
    Creates a DataFrame from YOLO prediction files in a folder.

    Args:
        predictions_folder (str): Path to the folder containing YOLO prediction files (.txt).
        img_size (Tuple[int, int]): Image size (width, height). Defaults to (1280, 720).

    Returns:
        pd.DataFrame: DataFrame with columns: image_id, category_id, norm_cx, norm_cy, norm_w, norm_h, bbox_x1, bbox_y1, bbox_x2, bbox_y2.
    """
    data = []
    img_width, img_height = img_size

    # Iterate through all .txt files in the folder
    for file_name in os.listdir(predictions_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(predictions_folder, file_name)
            image_id = file_name.replace(".txt", ".jpg")  # Convert .txt to .jpg
            
            # Read the YOLO prediction file
            with open(file_path, "r") as f:
                for line in f:
                    # Parse the YOLO prediction row
                    class_id, norm_cx, norm_cy, norm_w, norm_h, confidence = map(float, line.strip().split())
                    
                    # Convert normalized coordinates to absolute coordinates
                    cx, cy, w, h = norm_cx * img_width, norm_cy * img_height, norm_w * img_width, norm_h * img_height
                    x1, y1 = cx - w / 2, cy - h / 2
                    x2, y2 = cx + w / 2, cy + h / 2

                    # Add the row to the data list
                    data.append({
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "norm_cx": norm_cx,
                        "norm_cy": norm_cy,
                        "norm_w": norm_w,
                        "norm_h": norm_h,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "pred_conf": confidence
                    })

    # Create the DataFrame
    df = pd.DataFrame(data)
    return df
