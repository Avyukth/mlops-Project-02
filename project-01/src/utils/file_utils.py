import os
import yaml
import joblib

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, model_name, directory='models'):
    """
    Save model to disk.
    
    Args:
        model: The trained model
        model_name (str): Name of the model
        directory (str): Directory to save the model
    """
    ensure_dir(directory)
    joblib.dump(model, os.path.join(directory, f"{model_name}.joblib"))

def load_model(model_name, directory='models'):
    """
    Load model from disk.
    
    Args:
        model_name (str): Name of the model
        directory (str): Directory where the model is saved

    Returns:
        The loaded model
    """
    return joblib.load(os.path.join(directory, f"{model_name}.joblib"))

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
