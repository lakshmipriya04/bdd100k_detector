import argparse
import os

from dotenv import load_dotenv

from bdd_detector.bdd100k_detector import BDD100KDetection
from bdd_trainer import Trainer
from bdd_utils.model_utils import load_training_arguments_from_cmd\
    ,load_training_arguments_from_env

# Load environment variables
load_dotenv()


def generate_mappings(bdd100k_detection: BDD100KDetection):
    """
    Generate label-to-index and index-to-label mappings.

    Args:
        bdd100k_detection (BDD100KDetection): Initialized BDD100KDetection instance.
    """
    print("Generating label-to-index and index-to-label mappings...")
    label_to_idx = bdd100k_detection.labels2idx()
    idx_to_label = bdd100k_detection.idx2labels()
    print("Label-to-Index Mapping:", label_to_idx)
    print("Index-to-Label Mapping:", idx_to_label)


def add_bbox_attributes(bdd100k_detection: BDD100KDetection):
    """
    Add bounding box attributes and normalized values.

    Args:
        bdd100k_detection (BDD100KDetection): Initialized BDD100KDetection instance.
    """
    print("Adding bounding box attributes and normalized coordinates...")
    bdd100k_detection.add_bbox_attributes()


def create_dataloaders(bdd100k_detection: BDD100KDetection):
    """
    Create DataLoaders for training and validation splits.

    Args:
        bdd100k_detection (BDD100KDetection): Initialized BDD100KDetection instance.
        train_args (dict): Training arguments.

    Returns:
        tuple: Train and validation DataLoaders.
    """
    print("Creating DataLoaders...")
    train_loader = bdd100k_detection.create_dataloader(
        split="train",
        batch_size=bdd100k_detection.args.batch,
        shuffle=bdd100k_detection.args.shuffle,
        transform=None
    )
    val_loader = bdd100k_detection.create_dataloader(
        split="val",
        batch_size=bdd100k_detection.args.batch,
        shuffle=bdd100k_detection.args.shuffle,
        transform=None
    )
    print(f"Train DataLoader: {len(train_loader)} batches")
    print(f"Validation DataLoader: {len(val_loader)} batches")
    return train_loader, val_loader


def visualize_samples(bdd100k_detection: BDD100KDetection, output_path: str):
    """
    Visualize a few samples from the dataset.

    Args:
        bdd100k_detection (BDD100KDetection): Initialized BDD100KDetection instance.
        output_path (str): Path to save visualizations.
    """
    print("Visualizing training samples...")
    bdd100k_detection.visualize_samples(
        split='val',
        indices=list(range(8)),  # Display the first 8 samples
        display_image=False,
        save_path=os.path.join(output_path, "visualization.png"),
    )


def prepare_yolo_files(bdd100k_detection: BDD100KDetection):
    """
    Prepare YOLO-specific configuration and label files.

    Args:
        bdd100k_detection (BDD100KDetection): Initialized BDD100KDetection instance.
    """
    print("Preparing YOLO configuration and label files...")
    bdd100k_detection.create_yolo_config(bdd100k_detection.args.yolo_config_path)
    bdd100k_detection.create_labels(force=True)
    print("Checking YOLO label files...")
    bdd100k_detection.check_yolo_labels(10, 'visuals')


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BDD100K dataset custom arguments.")

    # Define arguments
    parser.add_argument('--runs_dir', type=str, default=os.getenv("RUNS_DIR", "./runs"),
                        help="Directory to save run outputs.")
    parser.add_argument('--num_epochs', type=int, default=int(os.getenv("NUM_EPOCHS", 50)), help="Number of epochs.")
    parser.add_argument('--imgsz', type=int, default=int(os.getenv("IMGSZ", 416)), help="Image size for training.")
    parser.add_argument('--resume', type=str, default=os.getenv("RESUME", None),
                        help="Path to a checkpoint to resume from.")
    parser.add_argument('--patience', type=int, default=int(os.getenv("PATIENCE", 10)), help="Early stopping patience.")
    parser.add_argument('--batch', type=int, default=int(os.getenv("BATCH", 16)), help="Batch size.")
    parser.add_argument('--conf_threshold', type=float, default=float(os.getenv("CONFIDENCE_THRESHOLD", 0.3)),
                        help="Confidence threshold for predictions.")
    parser.add_argument('--save_period', type=int, default=int(os.getenv("SAVE_PERIOD", 5)),
                        help="How often to save the model.")
    parser.add_argument('--project', type=str, default=os.getenv("PROJECT", "my_project"), help="Project name.")
    parser.add_argument('--fraction', type=float, default=float(os.getenv("FRACTION", 1.0)),
                        help="Fraction of the data to use.")
    parser.add_argument('--warmup_epochs', type=int, default=int(os.getenv("WARMUP_EPOCHS", 3)),
                        help="Number of warmup epochs.")
    parser.add_argument('--plots', type=bool, default=os.getenv("PLOTS", "True").lower() == "true",
                        help="Whether to plot training statistics.")
    parser.add_argument('--shuffle', type=bool, default=os.getenv("SHUFFLE", "True").lower() == "true",
                        help="Whether to shuffle data.")
    parser.add_argument('--images_save_path', type=str, default=os.getenv("IMAGES_SAVE_PATH", "./images"),
                        help="Path to save images.")
    parser.add_argument('--data_config_path', type=str, default=os.getenv("DATA_CONFIG_PATH", "./data.yaml"),
                        help="Path to dataset config.")
    parser.add_argument('--yolo_config_path', type=str, default=os.getenv("YOLO_CONFIG_PATH", "./yolo.yaml"),
                        help="Path to YOLO config.")
    parser.add_argument('--experiment_name', type=str, default=os.getenv("EXPERIMENT_NAME", "experiment_1"),
                        help="Experiment name.")
    parser.add_argument('--train', action='store_true', help="Enable the training (default: False)")
    parser.add_argument('--cmd', action='store_true', help="Accept command line arguments)")

    return parser.parse_args()


def main():
    # Assuming TrainingArguments is defined
    args = parse_args()
    training_args = load_training_arguments_from_env()
    # Comment if we want to load from env
    if args.cmd:
        training_args = load_training_arguments_from_cmd(args)
    project_location = os.environ.get('PROJECT_ROOT', os.path.abspath('../'))
    model_download_path = os.path.join(project_location, 'models')
    # Initialize the Trainer
    trainer = Trainer(training_args, model_name=f"{model_download_path}/yolo11s.pt")
    # Step 1: Initialize BDD100KDetection
    bdd100k_detection = BDD100KDetection(training_args,None,trainer)

    # Step 2: Generate label-to-index and index-to-label mappings
    generate_mappings(bdd100k_detection)

    # Step 3: Add bounding box attributes
    add_bbox_attributes(bdd100k_detection)

    # Step 4: Create DataLoaders
    train_loader, val_loader = create_dataloaders(bdd100k_detection)

    # Step 5: Visualize samples
    output_path = "visuals"
    os.makedirs(output_path, exist_ok=True)
    visualize_samples(bdd100k_detection, output_path)

    # Step 6: Prepare YOLO configuration and label files
    prepare_yolo_files(bdd100k_detection)
    if args.train:
        trainer.train()
    print("Run complete!")


if __name__ == "__main__":
    main()
