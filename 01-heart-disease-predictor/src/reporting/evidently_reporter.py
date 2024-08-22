import os
from datetime import datetime
from typing import Tuple, List

import mlflow
import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_preset import DataQualityTestPreset, DataStabilityTestPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
from src.utils.s3_utils import S3Utils

class EvidentlyReporter:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.PATHS["reports_dir"]
        self.s3_utils = S3Utils(config)
        self.s3_report_prefix = os.getenv("S3_REPORT_PREFIX", "evidently_reports")

    def generate_reports(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[str, str, str, str]:
        # Combine features and target for each dataset
        train_data = pd.concat([x_train, y_train], axis=1)
        test_data = pd.concat([x_test, y_test], axis=1)

        # Check for missing columns
        missing_columns = self._check_missing_columns(train_data, test_data)
        if missing_columns:
            print(f"Warning: The following columns are missing: {', '.join(missing_columns)}")

        # Generate unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evidently_report_{timestamp}.html"
        test_suite_filename = f"evidently_test_suite_{timestamp}.html"

        # Create an Evidently Report
        report = Report(
            metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()]
        )

        report.run(reference_data=train_data, current_data=test_data)
        report_path = os.path.join(self.output_dir, report_filename)
        report.save_html(report_path)

        # Create an Evidently Test Suite
        test_suite = TestSuite(
            tests=[
                DataStabilityTestPreset(),
                DataQualityTestPreset(),
            ]
        )

        # Add TestColumnDrift only for existing columns
        for column in ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure']:
            if column in train_data.columns and column in test_data.columns:
                test_suite._add_test(TestColumnDrift(column_name=column))
            else:
                print(f"Warning: Column '{column}' not found. Skipping drift test for this column.")

        test_suite.run(reference_data=train_data, current_data=test_data)
        test_suite_path = os.path.join(self.output_dir, test_suite_filename)
        test_suite.save_html(test_suite_path)

        # Upload reports to S3
        s3_report_key = os.path.join(self.s3_report_prefix, report_filename)
        s3_test_suite_key = os.path.join(self.s3_report_prefix, test_suite_filename)

        self.s3_utils.upload_file(report_path, s3_report_key)
        self.s3_utils.upload_file(test_suite_path, s3_test_suite_key)

        print(f"Evidently reports saved locally to {report_path} and {test_suite_path}")
        print(
            f"Evidently reports uploaded to S3 with keys {s3_report_key} and {s3_test_suite_key}"
        )

        return report_path, test_suite_path

    def _check_missing_columns(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> List[str]:
        """Check for columns that are missing in either train or test data."""
        train_columns = set(train_data.columns)
        test_columns = set(test_data.columns)
        return list(train_columns.symmetric_difference(test_columns))

    def log_reports_to_mlflow(
        self,
        report_path: str,
        test_suite_path: str,
        s3_report_key: str,
        s3_test_suite_key: str,
    ):
        mlflow.log_artifact(report_path, "evidently_reports")
        mlflow.log_artifact(test_suite_path, "evidently_reports")
        mlflow.log_param("evidently_report_s3_key", s3_report_key)
        mlflow.log_param("evidently_test_suite_s3_key", s3_test_suite_key)
