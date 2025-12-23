import pandas as pd  # to handle dataframes
from sklearn.model_selection import train_test_split  # to split data
from sklearn.ensemble import RandomForestRegressor  # to create model
from sklearn.preprocessing import OneHotEncoder  # to handle categorical variables
from sklearn.compose import ColumnTransformer  # to apply transformations
from sklearn.pipeline import Pipeline  # to create pipeline
import joblib  # to save model


def train_model(df):
    """
    Train a machine learning model using the provided dataframe
    and return the trained pipeline.
    """

    # Separate features (X) and target (y)
    X = df.drop('tip', axis=1)
    y = df['tip']

    # ✅ Explicitly define categorical and numerical columns
    categorical_cols = ['sex', 'smoker', 'day', 'time']
    numerical_cols = ['total_bill', 'size']

    # One-hot encode categorical features and pass numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    # Create pipeline with preprocessing + model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Return trained pipeline
    return model
