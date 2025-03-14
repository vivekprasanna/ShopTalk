import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    # You can perform all kinds of EDA in ML cycle here before passing this data to the model

    def merge_dataframes(self):
        df_products = pd.read_csv(self.config.data_path)
        df_products.rename(columns={"main_image_id": "image_id"}, inplace=True)
        df_images = pd.read_csv(self.config.images_path)
        df_images.rename(columns={"path": "image_path"}, inplace=True)
        df_product_image_merged = pd.merge(df_products, df_images, on="image_id", how="outer")
        df_product_image_merged.to_csv(self.config.data_master_path, index=False)

    def clean_merged_data(self):
        df_product_image_merged = pd.read_csv(self.config.data_master_path)
        logger.info(f"Shape of merged products: {df_product_image_merged.shape}")
        logger.info(f"Null values count by columns: {df_product_image_merged.isna().sum()}")
        # Item ID is the most important identifier - we cannot retrieve any product without item id.
        # So we will drop anything with Item ID as null
        df_product_image_merged.dropna(subset=['item_id'], inplace=True)
        logger.info(f"Shape of merged products: {df_product_image_merged.shape}")
        df_product_image_merged["enhanced_product_desc"] = "Given Product description: " \
                 + df_product_image_merged["product_description"].fillna("").astype(str) \
                 + ", " + df_product_image_merged["bullet_point"].fillna("").astype(str) \
                 + ", brand: " + df_product_image_merged["brand"].fillna("").astype(str) \
                 + ", weight: " + df_product_image_merged["item_weight"].fillna("").astype(str) \
                 + ", color: " + df_product_image_merged["color"].fillna("").astype(str) \
                 + ", height: " + df_product_image_merged["height"].fillna("").astype(str) \
                 + ", width: " + df_product_image_merged["width"].fillna("").astype(str) \
                 + ", model year: " + df_product_image_merged["model_year"].fillna("").astype(str) \
                 + ", shape: " + df_product_image_merged["item_shape"].fillna("").astype(str) \
                 + ", style: " + df_product_image_merged["style"].fillna("").astype(str) \
                 + ", material: " + df_product_image_merged["material"].fillna("") \
                 + ", product_type: " + df_product_image_merged["product_type"].fillna("").astype(str)
        ## Dropping null item name rows - as the product recommendation should give a product name
        df_product_image_merged.dropna(subset=['item_name'], inplace=True)
        logger.info(f"Shape of merged products after dropping null values in item_name: {df_product_image_merged.shape}")
        df_product_image_merged.to_csv(self.config.data_master_path, index=False)

    def create_sample_data(self):
        df_merged_master = pd.read_csv(self.config.data_master_path)
        selected_columns = ['item_id', 'item_name', 'product_type', 'country', 'enhanced_product_desc', 'image_path']

        df_sample = df_merged_master[selected_columns].sample(20000)
        df_sample.reset_index(drop=True, inplace=True)

        ## Drop duplicates
        df_sample.drop_duplicates()

        logger.info(f"Shape of sample dataset: {df_sample.shape}")
        df_sample.to_csv(self.config.data_sample_path, index=False)

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
