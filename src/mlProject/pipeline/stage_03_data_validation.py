from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import DataValidation
from mlProject import logger
from pathlib import Path
import os

STAGE_NAME = "Data Transformation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        if not os.path.exists(data_validation_config.data_sample_path):
            try:
                data_validation.merge_dataframes()
                data_validation.clean_merged_data()
                data_validation.create_sample_data()
                data_validation.create_image_captioning()
            except Exception as e:
                logger.exception(e)
                raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
