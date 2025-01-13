import os
from typing import Optional, Union


class TrainingArguments:
    def __init__(
            self,
            runs_dir: str,
            num_epochs: int,
            imgsz: int,
            resume: Optional[str],
            patience: int,
            batch: Union[int, float],
            save_period: int,
            project: str,
            fraction: float,
            warmup_epochs: int,
            plots: bool,
            shuffle: bool,
            conf_threshold: float,
            images_save_path: str,
            data_config_path: str,
            yolo_config_path: str,
            experiment_name: str
    ):
        self.runs_dir = runs_dir
        self.num_epochs = num_epochs
        self.imgsz = imgsz
        self.resume = resume
        self.patience = patience
        self.batch = batch
        self.conf_threshold = conf_threshold
        self.save_period = save_period
        self.project = os.path.join(runs_dir,project)
        self.fraction = fraction
        self.warmup_epochs = warmup_epochs
        self.plots = plots
        self.shuffle = shuffle
        self.images_save_path = images_save_path
        self.data_config_path = data_config_path
        self.yolo_config_path = yolo_config_path
        self.experiment_name = experiment_name
