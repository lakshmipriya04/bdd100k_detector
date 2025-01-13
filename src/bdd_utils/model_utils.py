import json
import os

from bdd_detector import TrainingArguments


def load_training_arguments_from_env() -> TrainingArguments:
    """Load TrainingArguments from environment variables."""
    return TrainingArguments(
        runs_dir=os.getenv("RUNS_DIR", "./runs"),
        num_epochs=int(os.getenv("NUM_EPOCHS", 50)),
        imgsz=int(os.getenv("IMGSZ", 416)),
        resume=os.getenv("RESUME", None),
        patience=int(os.getenv("PATIENCE", 10)),
        batch=int(os.getenv("BATCH", 16)),
        conf_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", 0.3)),
        save_period=int(os.getenv("SAVE_PERIOD", 5)),
        project=os.getenv("PROJECT", "my_project"),
        fraction=float(os.getenv("FRACTION", 1.0)),
        warmup_epochs=int(os.getenv("WARMUP_EPOCHS", 3)),
        plots=os.getenv("PLOTS", "True").lower() == "true",
        shuffle=os.getenv("SHUFFLE", "True").lower() == "true",
        images_save_path=os.getenv("IMAGES_SAVE_PATH", "./images"),
        data_config_path=os.getenv("DATA_CONFIG_PATH", "./data.yaml"),
        yolo_config_path=os.getenv("YOLO_CONFIG_PATH", "./yolo.yaml"),
        experiment_name=os.getenv("EXPERIMENT_NAME", f"experiment_1")
    )


def load_training_arguments_from_json(config_path: str) -> TrainingArguments:
    """Load TrainingArguments from a JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    return TrainingArguments(
        runs_dir=config.get("runs_dir", "./runs"),
        num_epochs=config.get("num_epochs", 50),
        imgsz=config.get("imgsz", 640),
        resume=config.get("resume", None),
        patience=config.get("patience", 10),
        batch=config.get("batch", 16),
        conf_threshold=config.get("conf_thresh", 0.25),
        save_period=config.get("save_period", 5),
        project=config.get("project", "my_project"),
        fraction=config.get("fraction", 1.0),
        warmup_epochs=config.get("warmup_epochs", 3),
        plots=config.get("plots", True),
        shuffle=config.get("shuffle", True),
        images_save_path=config.get("images_save_path", "./images"),
        data_config_path=config.get("data_config_path", "./data.yaml"),
        yolo_config_path=config.get("yolo_config_path", "./yolo.yaml"),
        experiment_name=config.get("experiment_name", "exp_1")
    )


def load_training_arguments_from_cmd(args) -> TrainingArguments:
    training_args = TrainingArguments(
        runs_dir=args.runs_dir,
        num_epochs=args.num_epochs,
        imgsz=args.imgsz,
        resume=args.resume,
        patience=args.patience,
        batch=args.batch,
        conf_threshold=args.conf_threshold,
        save_period=args.save_period,
        project=args.project,
        fraction=args.fraction,
        warmup_epochs=args.warmup_epochs,
        plots=args.plots,
        shuffle=args.shuffle,
        images_save_path=args.images_save_path,
        data_config_path=args.data_config_path,
        yolo_config_path=args.yolo_config_path,
        experiment_name=args.experiment_name
    )
    return training_args