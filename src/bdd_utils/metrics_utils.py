import numpy as np
import pandas as pd


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0


def evaluate_predictions(predictions_df:pd.DataFrame, val_df:pd.DataFrame, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth and compute TP, FP, FN for each category.
    """
    # Initialize columns for TP and matched annotation IDs
    predictions_df["TP"] = False
    predictions_df["annot_id"] = np.nan

    # Initialize a dictionary to track TP, FP, FN per category
    all_categories = set(predictions_df["category"].unique()).union(set(val_df["category"].unique()))
    category_metrics = {category: {"TP": 0, "FP": 0, "FN": 0} for category in all_categories}

    # Group by image_id for matching predictions with ground truths
    for image_id, pred_group in predictions_df.groupby("image_id"):
        gt_group = val_df[val_df["image_id"] == image_id].copy()

        # Create a column to track matched ground truth
        gt_group["matched"] = False

        for pred_idx, pred_row in pred_group.iterrows():
            pred_box = [pred_row["bbox_x1"], pred_row["bbox_y1"], pred_row["bbox_x2"], pred_row["bbox_y2"]]
            pred_label = pred_row["category"]

            # Compute IoU with all ground truth boxes of the same class
            gt_boxes = gt_group[gt_group["category"] == pred_label]
            ious = gt_boxes.apply(
                lambda gt_row: calculate_iou(
                    pred_box, [gt_row["bbox_x1"], gt_row["bbox_y1"], gt_row["bbox_x2"], gt_row["bbox_y2"]]
                ),
                axis=1,
            )

            # Find the best match
            if not ious.empty and ious.max() >= iou_threshold:
                best_match_idx = ious.idxmax()
                if not gt_group.loc[best_match_idx, "matched"]:  # Only match unmatched ground truth
                    predictions_df.at[pred_idx, "TP"] = True
                    predictions_df.at[pred_idx, "annot_id"] = gt_group.loc[best_match_idx, "annotation_id"]
                    gt_group.loc[best_match_idx, "matched"] = True

                    # Update TP count for the respective category
                    category_metrics[pred_label]["TP"] += 1
                else:
                    # Update FP count for the respective category
                    category_metrics[pred_label]["FP"] += 1
            else:
                # Update FP count for the respective category if no match found
                category_metrics[pred_label]["FP"] += 1

        # Update FN count for categories that have unmatched ground truth
        for _, gt_row in gt_group.iterrows():
            if not gt_row["matched"]:
                category_metrics[gt_row["category"]]["FN"] += 1

    return category_metrics, predictions_df


def calculate_category_metrics(category_metrics):
    """
    Calculate precision, recall, and F1 score for each category.
    """
    category_results = {}

    for category, metrics in category_metrics.items():
        tp = metrics["TP"]
        fp = metrics["FP"]
        fn = metrics["FN"]

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        category_results[category] = {"Precision": precision, "Recall": recall, "F1": f1}

    return category_results





