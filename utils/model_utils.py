import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, dict): # type: ignore
    """
    Preprocess the data by handling missing values, encoding categorical variables, and optionally normalizing the data.

    Args:
        df (pd.DataFrame): Data to preprocess.
        normalize (bool): If True, normalize the data.

    Returns:
        pd.DataFrame: Preprocessed data.
        dict: Label encoders for categorical variables.
    """
    # Drop customerID column
    df = df.drop('customerID', axis=1)
    
    # Replace spaces in column names with underscores
    df.columns = [col.replace(' ', '_') for col in df.columns]
    
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, label_encoders, scaler

def save_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, label_encoders: dict, scaler):
    """
    Save preprocessed data and label encoders to CSV files and a pickle file.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        label_encoders (dict): Label encoders for categorical variables.
    """
    X_train.to_csv('data/train/X_train.csv', index=False)
    X_test.to_csv('data/test/X_test.csv', index=False)
    y_train.to_csv('data/train/y_train.csv', index=False)
    y_test.to_csv('data/test/y_test.csv', index=False)
    
    # Saving label encoder
    joblib.dump(label_encoders, 'model/label_encoder/label_encoders.pkl')
    # Save scaler
    joblib.dump(scaler, 'model/scaler/scaler.pkl')

def load_preprocessed_data() -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series): # type: ignore
    """
    Load preprocessed data from CSV files.

    Returns:
        pd.DataFrame: Training features.
        pd.DataFrame: Testing features.
        pd.Series: Training labels.
        pd.Series: Testing labels.
    """
    X_train = pd.read_csv('data/train/X_train.csv')
    X_test = pd.read_csv('data/test/X_test.csv')
    y_train = pd.read_csv('data/train/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/test/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def save_model(model, model_path: str):
    """
    Save the trained model to a file.

    Args:
        model: Trained model.
        model_path (str): Path to save the model.
    """
    joblib.dump(model, model_path)

def load_model(model_path: str):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        Trained model.
    """
    return joblib.load(model_path)
