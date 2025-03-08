# Water Quality Multi-Task Prediction

## Project Overview  
This project implements a **Deep Learning-based multi-task model** to predict **Water Quality Index (WQI) (Regression)** and **Water Quality Classification (Classification)** using the **CPCB water quality dataset** in **PyTorch**. Instead of separate models, a **single neural network** is trained to predict both tasks. The model is evaluated using R² Score, Accuracy, and F1 Score.

## Introduction  
Access to clean water is critical, but water quality varies across regions due to environmental and human factors. This project predicts water quality based on chemical and physical parameters from India's **Central Pollution Control Board (CPCB) dataset**. The model simultaneously predicts:  

1. **Water Quality Index (WQI)** – A numerical measure of overall water quality.  
2. **Water Quality Classification** – A categorical label (e.g., Good, Poor, Unsuitable for Drinking).  

Using **Deep Learning**, this multi-task model helps in water monitoring and pollution control.

## How It Works  

1. **Data Processing**:  
   - The dataset is preprocessed, handling missing values.  
   - Features include chemical & physical properties such as **pH, TDS, Hardness, etc.**  
   
2. **Multi-Task Learning Model**:  
   - A **single neural network** with shared layers predicts both **WQI (Regression)** and **Water Quality Classification (Classification)** simultaneously.  
   - The model is trained using **loss functions for both tasks**.  

3. **Evaluation Metrics**:  
   - **Regression**: R² Score  
   - **Classification**: Accuracy & F1 Score  

## Installation  

To set up the project, install the required dependencies:  

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn
```

## Results
Final Validation Results After Training:

- Best Validation R² Score: 1.0000
- Best Validation Accuracy: 0.9116
- Best Validation F1 Score: 0.9118

Final Test Results:

- Regression R² Score: 1.0000
- Classification Accuracy: 0.9163
- Classification F1 Score: 0.9163

## Conclusion
This project demonstrates the power of multi-task learning in predicting both numerical and categorical water quality measures efficiently. The model achieves high accuracy and helps in better water monitoring.
