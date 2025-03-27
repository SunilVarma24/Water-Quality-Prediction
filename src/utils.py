# src/utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def plot_class_distribution(y_class_true):
    plt.figure(figsize=(8, 5))
    unique_classes, class_counts = np.unique(y_class_true, return_counts=True)
    sns.barplot(x=unique_classes, y=class_counts, palette='viridis')
    plt.xlabel("Water Quality Category")
    plt.ylabel("Count")
    plt.title("Class Distribution in Dataset")
    plt.xticks(rotation=45)
    plt.show()

def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def plot_classification_report_heatmap(y_true, y_pred):
    class_report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(class_report).iloc[:-1, :].T
    plt.figure(figsize=(8, 5))
    sns.heatmap(report_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Classification Report Heatmap")
    plt.show()

def plot_regression_residuals_hist(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color='blue')
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution (Regression)")
    plt.show()

def plot_residuals_scatter(y_pred, y_true):
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted Values")
    plt.show()
