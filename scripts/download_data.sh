#!/bin/bash

# URLs to download
URL1="https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip"
URL2="https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip"
URL3="https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip"

# Default directory if user doesn't specify
default_dir="$HOME/bdd100k_data"

# Check if the destination path is provided as a parameter
dest_path="${1:-$default_dir}"

# Check if the directory exists, if not, create it
if [ ! -d "$dest_path" ]; then
  echo "Directory does not exist. Creating directory $dest_path..."
  mkdir -p "$dest_path"
fi

# Download the files in parallel
echo "Downloading 100k_images_val.zip..."
wget -P "$dest_path" $URL1 &

echo "Downloading 100k_images_train.zip..."
wget -P "$dest_path" $URL2 &

echo "Downloading bdd100k_det_20_labels_trainval.zip..."
wget -P "$dest_path" $URL3 &

# Wait for all downloads to finish
wait

# Unzip the downloaded files
echo "Unzipping 100k_images_val.zip..."
unzip -q "$dest_path/100k_images_val.zip" -d "$dest_path"

echo "Unzipping 100k_images_train.zip..."
unzip -q "$dest_path/100k_images_train.zip" -d "$dest_path"

echo "Unzipping bdd100k_det_20_labels_trainval.zip..."
unzip -q "$dest_path/bdd100k_det_20_labels_trainval.zip" -d "$dest_path"

# Delete the zip files after unzipping
echo "Deleting zip files..."
rm "$dest_path/100k_images_val.zip"
rm "$dest_path/100k_images_train.zip"
rm "$dest_path/bdd100k_det_20_labels_trainval.zip"

echo "All downloads, unzips, and clean-up are complete!"
