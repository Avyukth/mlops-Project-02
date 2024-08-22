import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_preprocessor import FeaturePreprocessor
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config, s3_utils):
        self.config = config
        self.s3_utils = s3_utils
        self.preprocessor = None
        self.feature_names = None

    def load_data(self):
        data_path = self.config.paths['data_dir']
        df = pd.read_csv(data_path)
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        return X, y

    def load_and_preprocess_data(self):
        X, y = self.load_data()

        self.preprocessor = Pipeline([
            ('feature_preprocessor', FeaturePreprocessor()),
            ('scaler', StandardScaler())
        ])

        X_processed = self.preprocessor.fit_transform(X, y)

        self.feature_names = self.preprocessor.named_steps['feature_preprocessor'].get_feature_names_out()

        # Convert X_processed back to a DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=self.feature_names, index=X.index)

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        return X_processed_df, y, cat_cols, num_cols

    def save_preprocessors(self, X):
        preprocessor_path = os.path.join(self.config.paths['model_dir'], "preprocessor.onnx")
        
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onx = convert_sklearn(self.preprocessor, initial_types=initial_type)
        onnx.save_model(onx, preprocessor_path)

        logger.info(f"Preprocessor saved to {preprocessor_path}")

    def split_data(self, X, y):
        return train_test_split(
            X, y, 
            test_size=self.config.train_test_split['test_size'], 
            random_state=self.config.train_test_split['random_state']
        )

    def load_preprocessors(self):
        preprocessor_path = os.path.join(self.config.paths['model_dir'], "preprocessor.onnx")
        
        self.s3_utils.download_file("preprocessor.onnx", preprocessor_path)
        
        self.preprocessor = onnx.load(preprocessor_path)
        
        logger.info("Preprocessor loaded from S3 successfully")
