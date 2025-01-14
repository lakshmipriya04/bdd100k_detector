import fiftyone as fo
import fiftyone.utils.yolo as fouy

name = "bdd100k"
dataset_dir = "/home/jl_fs/bdd100k_detector/data/raw/bdd100k/"

# The splits to load
splits = ["val"]

# Load the dataset, using tags to mark the samples in each split
dataset = fo.Dataset(name,persistent=True,overwrite=True)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
        labels_path=f'/home/jl_fs/bdd100k_detector/data/raw/bdd100k/labels/100k/{split}',
        label_field="ground_truth"
    )
classes = ['bicycle', 'bus', 'car', 'motorcycle', 'other person', 'other vehicle', 'pedestrian', 'rider',
           'traffic light', 'traffic sign', 'trailer', 'train', 'truck']
fouy.add_yolo_labels(
    dataset,
    "predictions",
    "/home/jl_fs/bdd100k_detector/runs/bdd100_detector/validation_predictions/labels",
    classes
)

results = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="eval",
)
# Print a classification report for the top-10 classes
results.print_report(classes=classes)
session = fo.launch_app(dataset)
session.wait()
