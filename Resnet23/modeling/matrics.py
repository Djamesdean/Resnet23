import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
import torch


def evaluate_metrics(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    one_hot_labels = np.eye(num_classes)[all_labels]

    # Classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    for class_label, class_metrics in report.items():
        if isinstance(class_metrics, dict):  
            for metric_name, value in class_metrics.items():
                mlflow.log_metric(f"{class_label}_{metric_name}", value)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/confusion_matrix.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)

    # mAP 
    mAP = average_precision_score(one_hot_labels, all_probs, average="macro")
    mlflow.log_metric("mean_average_precision", mAP)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")