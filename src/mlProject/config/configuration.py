from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            DataVisualizationConfig,
                                            ModelTrainerConfig,
                                            ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH,
            schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            listings_path=config.listings_path,
            images_path=config.images_path,
            all_schema=schema,
            data_csv_path=config.data_csv_path,
        )

        return data_transformation_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            images_path=config.images_path,
            data_master_path=config.data_master_path,
            data_sample_path=config.data_sample_path,
            STATUS_FILE=config.STATUS_FILE,
            image_path_prefix=config.image_path_prefix
        )

        return data_validation_config

    def get_data_visualization_config(self) -> DataVisualizationConfig:
        config = self.config.data_visualization

        create_directories([config.root_dir])

        data_visualization_config = DataVisualizationConfig(
            root_dir=config.root_dir,
            data_master_path=config.data_master_path,
            data_sample_path=config.data_sample_path,
        )

        return data_visualization_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_name=config.model_name,
            data_path=config.data_path,
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            # mlflow_uri="https://dagshub.com/someshnaman/End_to_end_MLOPS_project.mlflow",
            mlflow_uri="https://dagshub.com/ashokj0922/End_to_end_mlops.mlflow",
        )

        return model_evaluation_config
