from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_visualization import DataVisualization
from mlProject import logger
import pandas as pd

STAGE_NAME = "Data Transformation stage"


class DataVisualizationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_visualization_config = config.get_data_visualization_config()
        df_merged_master = pd.read_csv(data_visualization_config.data_master_path)
        try:
            data_visualization = DataVisualization(config=data_visualization_config, df_merged_master=df_merged_master)
            data_visualization.generate_top_10_product_types()
            data_visualization.generate_top_10_product_brands()
            data_visualization.generate_number_products_per_country()
            data_visualization.generate_top_5_product_categories_by_country()
            data_visualization.generate_word_cloud_prod_description()
            data_visualization.generate_word_cloud_prod_names()
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataVisualizationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
