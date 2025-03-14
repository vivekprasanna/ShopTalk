import pandas as pd
import glob
import json
import os

from mlProject import logger

from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_and_save(self) -> bool:
        if not os.path.exists(self.config.data_csv_path):
            try:
                json_files = glob.glob(self.config.listings_path)
                dfs = []

                for f in json_files:
                    with open(f, 'r') as file:
                        for line in file:  # Read line by line to handle multiple JSON objects
                            data = json.loads(line.strip())  # Parse each JSON object separately

                            extracted_data = {}  # Store extracted values

                            # Loop through each key in the JSON object
                            for key, value in data.items():
                                if isinstance(value, list):  # Only process lists
                                    for item in value:
                                        if isinstance(item, dict) and "language_tag" in item and item["language_tag"].startswith("en_"):
                                            extracted_data[key] = item["value"]  # Store the corresponding value
                                        elif isinstance(item, dict) and "language_tag" not in item and "value" in item:
                                            extracted_data[key] = item["value"]
                                else:
                                    extracted_data[key] = value

                            # Convert extracted data into DataFrame
                            df = pd.DataFrame([extracted_data])
                            dfs.append(df)

                # Combine all DataFrames
                df_products = pd.concat(dfs, ignore_index=True)

                # all_cols = list(df_products.columns)

                # all_schema = self.config.all_schema.keys()
                #
                # for col in all_cols:
                #     if col not in all_schema:
                #         # validation_status = False
                #         with open(self.config.STATUS_FILE, 'w') as f:
                #             f.write(f"Additional Column found: {col}, may be ignored")

                logger.info("Saving data as csv under artifacts/data_validation/df_data.csv!")
                df_products.to_csv(self.config.data_csv_path, index=False)
                return True

            except Exception as e:
                raise e

        return True
