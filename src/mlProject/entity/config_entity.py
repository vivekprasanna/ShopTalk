from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    STATUS_FILE: str
    all_schema: dict
    listings_path: str
    images_path: str
    data_csv_path: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    images_path: str
    data_master_path: str
    data_sample_path: str
    STATUS_FILE: str
    image_path_prefix: str


@dataclass(frozen=True)
class DataVisualizationConfig:
    root_dir: Path
    data_master_path: str
    data_sample_path: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: str
    model_name: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
