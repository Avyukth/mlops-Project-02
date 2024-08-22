import os
from datetime import datetime
from typing import Tuple

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
        """
        Initialize the EvidentlyReporter.

        Args:
            config: Configuration object containing necessary settings.
        """
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
        """
        Generate Evidently Test Suite and Report, save locally and upload to S3.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            x_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.

        Returns:
            Tuple[str, str, str, str]: Paths to the generated report and test suite (local and S3).
        """
        # Combine features and target for each dataset
        train_data = pd.concat([x_train, y_train], axis=1)
        test_data = pd.concat([x_test, y_test], axis=1)

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
                TestColumnDrift(column_name="age"),
                TestColumnDrift(column_name="sex"),
                TestColumnDrift(column_name="chest_pain_type"),
                TestColumnDrift(column_name="resting_blood_pressure"),
            ]
        )

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

    def log_reports_to_mlflow(
        self,
        report_path: str,
        test_suite_path: str,
        s3_report_key: str,
        s3_test_suite_key: str,
    ):
        """
        Log the generated reports to MLflow.

        Args:
            report_path (str): Local path to the Evidently report.
            test_suite_path (str): Local path to the Evidently test suite.
            s3_report_key (str): S3 key for the Evidently report.
            s3_test_suite_key (str): S3 key for the Evidently test suite.
        """
        mlflow.log_artifact(report_path, "evidently_reports")
        mlflow.log_artifact(test_suite_path, "evidently_reports")
        mlflow.log_param("evidently_report_s3_key", s3_report_key)
        mlflow.log_param("evidently_test_suite_s3_key", s3_test_suite_key)
