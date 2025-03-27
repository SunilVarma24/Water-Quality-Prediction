# src/data.py

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

def load_and_preprocess_data(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Impute missing values in 'Village' using the most frequent village within the same (State, District, Block)
    def impute_village(group):
        most_frequent_village = group['Village'].mode()
        if not most_frequent_village.empty:
            group['Village'].fillna(most_frequent_village.iloc[0], inplace=True)
        return group

    df = df.groupby(['State', 'District', 'Block'], group_keys=False).apply(impute_village)
    # Check remaining missing values in Village (for debugging if needed)
    # print(df['Village'].isnull().sum())

    # Impute missing 'Block' values using the most frequent Block within the same State & District
    def impute_block(group):
        most_frequent_block = group['Block'].mode()
        if not most_frequent_block.empty:
            group['Block'].fillna(most_frequent_block.iloc[0], inplace=True)
        return group

    df = df.groupby(['State', 'District'], group_keys=False).apply(impute_block)
    # print(df['Block'].isnull().sum())

    # Impute missing Latitude & Longitude using mean values within the same State & District
    def impute_lat_lon(group):
        group['Latitude'].fillna(group['Latitude'].mean(), inplace=True)
        group['Longitude'].fillna(group['Longitude'].mean(), inplace=True)
        return group

    df = df.groupby(['State', 'District'], group_keys=False).apply(impute_lat_lon)
    # print(df[['Latitude', 'Longitude']].isnull().sum())

    # Drop rows where Latitude or Longitude is still missing
    df = df.dropna(subset=['Latitude', 'Longitude'])
    # Confirm no missing values remain (if needed)
    # print(df[['Latitude', 'Longitude']].isnull().sum())

    # Drop remaining duplicate rows
    df = df.drop_duplicates()
    # print(df.duplicated().sum())

    # Remove unwanted columns
    df.drop(columns=['Well_ID'], inplace=True)

    # Encode categorical variables
    categorical_cols = ['State', 'District', 'Block', 'Village']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Splitting data into features and targets
    X = df.drop(columns=['WQI', 'Water Quality Classification'])
    y_reg = df['WQI'].values  # Regression target
    y_class = df['Water Quality Classification'].astype('category').cat.codes.values  # Classification target

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X_scaled, y_reg, y_class, test_size=0.15, random_state=210
    )
    X_train, X_val, y_reg_train, y_reg_val, y_class_train, y_class_val = train_test_split(
        X_train, y_reg_train, y_class_train, test_size=0.1, random_state=210
    )
    
    print("Training set size:", len(X_train))
    print("Validation set size:", len(X_val))
    print("Test set size:", len(X_test))
    
    return (X_train, X_val, X_test, 
            y_reg_train, y_reg_val, y_reg_test, 
            y_class_train, y_class_val, y_class_test, scaler)

# Custom Dataset Class
class WaterQualityDataset(Dataset):
    def __init__(self, X, y_reg, y_class):
        self.X = X
        self.y_reg = y_reg
        self.y_class = y_class

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_class[idx]