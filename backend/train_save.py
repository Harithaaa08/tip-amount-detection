"""
The purpose of this file is to:
1. Load the data from load_data.py
2. Train the model using train_model.py
3. Save the trained model to a file using joblib
"""

import joblib  # used to save the trained model

from data.load_data import load_data  # function to load dataset
from model.train_model import train_model  # function to train model


def train_and_save_model():
    """
    Load data, train the model, and save the trained model to a file
    """

    # Load the dataset
    df = load_data()

    # Train the model
    model = train_model(df)

    # Save the trained model to a file
    joblib.dump(model, "random_forest_model.pkl")

    print("✅ Model trained and saved as random_forest_model.pkl")


# ✅ THIS MUST BE OUTSIDE THE FUNCTION
if __name__ == "__main__":
    train_and_save_model()
