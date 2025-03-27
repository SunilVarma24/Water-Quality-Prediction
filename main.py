# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, accuracy_score, f1_score
from src import config, data, model, utils

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. Load and preprocess data
(X_train, X_val, X_test, 
 y_reg_train, y_reg_val, y_reg_test, 
 y_class_train, y_class_val, y_class_test, scaler) = data.load_and_preprocess_data(config.DATA_PATH)

# 2. Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_reg_train_tensor = torch.tensor(y_reg_train, dtype=torch.float32).to(device)
y_reg_val_tensor = torch.tensor(y_reg_val, dtype=torch.float32).to(device)
y_reg_test_tensor = torch.tensor(y_reg_test, dtype=torch.float32).to(device)
y_class_train_tensor = torch.tensor(y_class_train, dtype=torch.long).to(device)
y_class_val_tensor = torch.tensor(y_class_val, dtype=torch.long).to(device)
y_class_test_tensor = torch.tensor(y_class_test, dtype=torch.long).to(device)

# 3. Create DataLoaders using the custom dataset
train_dataset = data.WaterQualityDataset(X_train_tensor, y_reg_train_tensor, y_class_train_tensor)
val_dataset = data.WaterQualityDataset(X_val_tensor, y_reg_val_tensor, y_class_val_tensor)
test_dataset = data.WaterQualityDataset(X_test_tensor, y_reg_test_tensor, y_class_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# 4. Initialize the multi-task model
input_dim = X_train.shape[1]
num_classes = len(np.unique(y_class_train))
net = model.MultiTaskNN(input_dim, num_classes).to(device)

# 5. Define loss functions and optimizer
criterion_reg = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

# 6. Training loop
epochs = config.EPOCHS
best_val_loss = float('inf')
best_val_accuracy = 0
best_val_f1 = 0
best_val_r2 = float('-inf')

val_accuracies = []
val_f1_scores = []
val_r2_scores = []

for epoch in range(epochs):
    net.train()
    total_loss, total_reg_loss, total_class_loss = 0, 0, 0

    for X_batch, y_reg_batch, y_class_batch in train_loader:
        optimizer.zero_grad()

        reg_pred, class_pred = net(X_batch)

        loss_reg = criterion_reg(reg_pred.squeeze(), y_reg_batch)
        loss_class = criterion_class(class_pred, y_class_batch)
        loss = loss_reg + loss_class  # Combined loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reg_loss += loss_reg.item()
        total_class_loss += loss_class.item()

    # Compute average training loss
    avg_train_loss = total_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)
    avg_class_loss = total_class_loss / len(train_loader)

    # Validation Phase
    net.eval()
    val_loss, val_reg_loss, val_class_loss = 0, 0, 0
    y_reg_preds, y_class_preds = [], []
    y_reg_true, y_class_true = [], []

    with torch.no_grad():
        for X_batch, y_reg_batch, y_class_batch in val_loader:
            reg_pred, class_pred = net(X_batch)

            loss_reg = criterion_reg(reg_pred.squeeze(), y_reg_batch)
            loss_class = criterion_class(class_pred, y_class_batch)
            loss = loss_reg + loss_class  # Combined loss

            val_loss += loss.item()
            val_reg_loss += loss_reg.item()
            val_class_loss += loss_class.item()

            y_reg_preds.extend(reg_pred.cpu().numpy())
            y_class_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
            y_reg_true.extend(y_reg_batch.cpu().numpy())
            y_class_true.extend(y_class_batch.cpu().numpy())

    # Compute average validation loss
    avg_val_loss = val_loss / len(val_loader)
    avg_val_reg_loss = val_reg_loss / len(val_loader)
    avg_val_class_loss = val_class_loss / len(val_loader)

    # Compute validation metrics
    y_reg_preds = np.array(y_reg_preds).flatten()
    y_class_preds = np.array(y_class_preds)
    val_r2 = r2_score(y_reg_true, y_reg_preds)
    val_accuracy = accuracy_score(y_class_true, y_class_preds)
    val_f1 = f1_score(y_class_true, y_class_preds, average='weighted')

    # Store metrics for plotting later if needed
    val_accuracies.append(val_accuracy)
    val_f1_scores.append(val_f1)
    val_r2_scores.append(val_r2)

    # Save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_r2 = val_r2
        best_val_accuracy = val_accuracy
        best_val_f1 = val_f1
        torch.save(net.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Loss: {avg_train_loss:.4f} (Reg: {avg_reg_loss:.4f}, Class: {avg_class_loss:.4f}) | "
          f"Val Loss: {avg_val_loss:.4f} (Reg: {avg_val_reg_loss:.4f}, Class: {avg_val_class_loss:.4f})")

print("\n-----------------------------------------------------\n")

# Print final validation metrics
print("\nFinal Validation Results:")
print(f"Best Validation R² Score: {best_val_r2:.4f}")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
print(f"Best Validation F1 Score: {best_val_f1:.4f}")

# Load best model for final test evaluation
net.load_state_dict(torch.load("best_model.pth"))

# 7. Final Test Evaluation
net.eval()
y_reg_preds, y_class_preds = [], []
y_reg_true, y_class_true = [], []

with torch.no_grad():
    for X_batch, y_reg_batch, y_class_batch in test_loader:
        reg_pred, class_pred = net(X_batch)

        y_reg_preds.extend(reg_pred.cpu().numpy())
        y_class_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
        y_reg_true.extend(y_reg_batch.cpu().numpy())
        y_class_true.extend(y_class_batch.cpu().numpy())

# Convert predictions to numpy arrays
y_reg_preds = np.array(y_reg_preds).flatten()
y_class_preds = np.array(y_class_preds)

# Compute test metrics
test_r2 = r2_score(y_reg_true, y_reg_preds)
test_accuracy = accuracy_score(y_class_true, y_class_preds)
test_f1 = f1_score(y_class_true, y_class_preds, average='weighted')

print(f"Final Test Results:\n"
      f"Regression R² Score: {test_r2:.4f}\n"
      f"Classification Accuracy: {test_accuracy:.4f}\n"
      f"Classification F1 Score: {test_f1:.4f}")

# 8. Plotting visualizations

# Class Distribution Bar Chart
utils.plot_class_distribution(y_class_true)

# Confusion Matrix
utils.plot_conf_matrix(y_class_true, y_class_preds)

# Classification Report Heatmap
utils.plot_classification_report_heatmap(y_class_true, y_class_preds)

# Regression Residual Analysis (Histogram)
utils.plot_regression_residuals_hist(y_reg_true, y_reg_preds)

# Scatter Plot of Residuals
utils.plot_residuals_scatter(y_reg_preds, y_reg_true)