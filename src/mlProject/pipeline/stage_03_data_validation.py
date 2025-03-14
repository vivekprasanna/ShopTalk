from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import DataValidation
from mlProject import logger
from pathlib import Path

STAGE_NAME = "Data Transformation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_validation_config = config.get_data_transformation_config()
                data_validation = DataValidation(config=data_validation_config)
                # data_transformation.train_test_spliting()

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
