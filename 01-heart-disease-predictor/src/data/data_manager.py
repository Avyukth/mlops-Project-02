import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config, s3_utils):
        self.config = config
        self.s3_utils = s3_utils
        self.preprocessor = None

    def load_and_preprocess_data(self):
        # Load your data here
        X, y = self.load_data()

        # Create and fit the preprocessing pipeline
        self.preprocessor = Pipeline([
            ('feature_preprocessor', FeaturePreprocessor()),
            ('scaler', StandardScaler())
        ])

        X_processed = self.preprocessor.fit_transform(X)

        # Get the column names after preprocessing
        feature_names = self.preprocessor.named_steps['feature_preprocessor'].get_feature_names_out()

        return X_processed, y, feature_names

    def save_preprocessors(self, X):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx

        preprocessor_path = os.path.join(self.config.PATHS['model_dir'], "preprocessor.onnx")
        
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onx = convert_sklearn(self.preprocessor, initial_types=initial_type)
        onnx.save_model(onx, preprocessor_path)

    def split_data(self, X, y):
        return train_test_split(
            X, y, 
            test_size=self.config.TRAIN_TEST_SPLIT['test_size'], 
            random_state=self.config.TRAIN_TEST_SPLIT['random_state']
        )

    def load_preprocessors(self):
        feature_preprocessor_path = os.path.join(self.config.MODEL_DIR, "feature_preprocessor.pkl")
        target_preprocessor_path = os.path.join(self.config.MODEL_DIR, "target_preprocessor.pkl")
        
        # Download preprocessors from S3
        self.s3_utils.download_file("feature_preprocessor.pkl", feature_preprocessor_path)
        self.s3_utils.download_file("target_preprocessor.pkl", target_preprocessor_path)
        
        with open(feature_preprocessor_path, 'rb') as f:
            self.preprocessor.feature_preprocessor = cloudpickle.load(f)
        with open(target_preprocessor_path, 'rb') as f:
            self.preprocessor.target_preprocessor = cloudpickle.load(f)
        logger.info("Preprocessors loaded from S3 successfully")
