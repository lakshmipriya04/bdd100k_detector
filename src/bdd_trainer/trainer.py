# Contains code for initialization of trainer, method for train and validation
import os
import shutil

import torch
from ultralytics import YOLO
from ultralytics import settings

from bdd_detector import TrainingArguments


class Trainer:
    def __init__(self, training_args: TrainingArguments, model_name: str):
        """
        Initialize the Trainer class.

        Args:
            training_args (TrainingArguments): An instance of the TrainingArguments class containing training configurations.
            model_name (str): Name or path of the YOLO model to be used.
        """
        self.training_args = training_args
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.device = "0" if torch.cuda.is_available() else "cpu"
        project_root = os.path.dirname(self.training_args.project)
        mlflow_path = os.path.join(project_root,'mlflow')
        os.makedirs(mlflow_path,exist_ok=True)
        settings.update({"mlflow": True})  # Enable MLflow logging

    def train(self):
        """
        Train the YOLO model using the specified training arguments.
        """
        train_results = self.model.train(
            data=self.training_args.yolo_config_path,  # Path to dataset YAML
            epochs=self.training_args.num_epochs,  # Number of training epochs
            imgsz=self.training_args.imgsz,  # Training image size
            device=self.device,  # GPU or CPU device
            plots=self.training_args.plots,
            batch=self.training_args.batch,
            multi_scale=True,
            patience=self.training_args.patience,
            save_period=self.training_args.save_period,
            project=self.training_args.project,
            name=self.training_args.experiment_name
        )
        try:
            checkpoint_path = os.path.join(self.training_args.project,self.training_args.experiment_name,'weights','best.pt')
            shutil.copy(checkpoint_path,os.environ.get('MODEL_PATH'))
        except:
            pass
        return train_results
    
    def tune(self):
        # Define search space
        search_space = {
            "lr0": (1e-5, 1e-1),
             "mosaic":(0.0,1.0),
             "mixup":(0.0,1.0)
        }

        # Tune hyperparameters on COCO8 for 30 epochs
        self.model.tune(
            data=self.training_args.yolo_config_path,
            epochs=5,
            imgsz=self.training_args.imgsz,  # Training image size
            device=self.device,
            iterations=300,
            project=self.training_args.project,
            name=self.training_args.experiment_name,
            optimizer="AdamW",
            space=search_space,
            plots=True,
            save=True,
            val=True,
        )
    
    def resume_train(self,checkpoint_path):
        self.model = YOLO(checkpoint_path)
        train_results = self.model.train(
            data=self.training_args.yolo_config_path,  # Path to dataset YAML
            epochs=self.training_args.num_epochs,  # Number of training epochs
            imgsz=self.training_args.imgsz,  # Training image size
            device=self.device,  # GPU or CPU device
            plots=self.training_args.plots,
            batch=self.training_args.batch,
            patience=self.training_args.patience,
            save_period=self.training_args.save_period,
            project=self.training_args.project,
            resume=True
        )
        return train_results

    def evaluate(self):
        """
        Evaluate the YOLO model on the validation set.
        """
        metrics = self.model.val(data=self.training_args.yolo_config_path, save_json=True,
                                 plots=self.training_args.plots)
        return metrics

    def predict(self, image_path: str, save: bool = False, conf: float = 0.25):
        """
        Run predictions on an image or batch of images.

        Args:
            image_path (str): Path to the image or directory of images for prediction.
            save (bool): Whether to save predictions.
            conf (float): Confidence threshold for predictions.

        Returns:
            Results of the prediction.
        """
        results = self.model.predict(source=image_path, save=save, conf=conf)
        return results

    def export_model(self, export_path: str, format: str = "torchscript"):
        """
        Export the trained YOLO model in a specific format.

        Args:
            export_path (str): Directory to save the exported model.
            format (str): Export format ('torchscript', 'onnx', etc.).

        Returns:
            Path to the exported model.
        """
        os.makedirs(export_path, exist_ok=True)
        export_path = os.path.join(export_path, f"model.{format}")
        self.model.export(format=format, dynamic=True)
        return export_path
