import os

import torch
from bdd_detector import TrainingArguments
from ultralytics import YOLO


class Evaluator:
    def __init__(self, args: TrainingArguments, model_name: str, checkpoint_path: str = None):
        """
        Initialize the Evaluator class.

        Args:
            args (TrainingArguments): An instance of the TrainingArguments class containing evaluation configurations.
            model_name (str): Name or path of the YOLO model to be used.
            checkpoint_path (str, optional): Path to the checkpoint file to load for evaluation. Defaults to None.
        """
        self.args = args
        self.model_name = model_name
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(checkpoint_path if checkpoint_path else model_name)
        self.results_dir = os.path.join(os.path.dirname(self.args.project), "evaluation_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate(self, save_json: bool = True, save_plots: bool = True):
        """
        Evaluate the YOLO model on the validation set.

        Args:
            save_json (bool): Whether to save the evaluation metrics in a JSON file.
            save_plots (bool): Whether to save the evaluation plots.

        Returns:
            dict: Evaluation metrics.
        """
        metrics = self.model.val(
            data=self.args.yolo_config_path,
            save_json=save_json,
            plots=save_plots
        )
        return metrics

    def test(self, data_path: str, save: bool = True, conf: float = 0.25, test_time_aug: bool = True,
             save_txt: bool = False, save_conf: bool = False):
        """
        Test the YOLO model on a specified test dataset.

        Args:
            data_path (str): Path to the test dir or test image file path.
            conf (float): Confidence threshold for predictions.

        Returns:
            list: Test results.
        """
        test_results = self.model.predict(
            source=data_path,
            save=save,
            conf=conf,
            imgsz=self.args.imgsz,
            device=self.device,
            project=self.args.project,
            augment=test_time_aug,
            name="output_predictions",
            save_txt=save_txt,
            save_conf=save_conf
        )
        return test_results

    def visualize_predictions(self, image_path: str, conf: float = 0.25, save: bool = True):
        """
        Visualize predictions on an image or batch of images.

        Args:
            image_path (str): Path to the image or directory of images for visualization.
            conf (float): Confidence threshold for predictions.
            save (bool): Whether to save the visualizations.

        Returns:
            Visualization results.
        """
        results = self.model.predict(source=image_path, conf=conf, save=save)
        return results
