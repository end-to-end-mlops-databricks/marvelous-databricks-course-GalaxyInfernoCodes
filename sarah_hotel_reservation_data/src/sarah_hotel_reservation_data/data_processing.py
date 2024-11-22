from typing import Tuple

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split


class HotelDataset:
    def __init__(self, data_filepath, yaml_file_path):
        self.full_data = self.load_dataset(data_filepath)
        # read yaml file into dictionaary
        self.config = self.load_config(yaml_file_path)

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load the dataset from the given file path.

        Args:
            filepath (str): Path to the dataset.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        df = pd.read_csv(filepath)
        return df

    def load_config(self, yaml_file_path: str) -> dict:
        """
        Load data processing configurations from a yaml file.

        Args:
            yaml_file_path (str): Path to the yaml file.

        Returns:
            dict: Data processing configurations.
        """
        with open(yaml_file_path, "r") as yaml_file:
            project_config = yaml.safe_load(yaml_file)
        return project_config

    def drop_columns(self) -> pd.DataFrame:
        """
        Drop the columns declared as "drop_columns" in the data config file from the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the column dropped.
        """
        columns_to_drop = self.config["drop_columns"]
        self.full_data = self.full_data.drop(columns=columns_to_drop)
        return self.full_data

    def convert_target_variable(self) -> pd.DataFrame:
        """
        Convert the target variable (defined via config) into binary.

        Returns:
            pd.DataFrame: DataFrame with target_variable as a binary column.
        """
        target_variable = self.config["target_variable"]
        self.full_data[target_variable] = (
            self.full_data[target_variable] == "Canceled"
        ).astype(int)

        return self.full_data

    def run_data_preparation(self) -> None:
        self.convert_target_variable()

        # Handle numeric features
        num_features = self.config["numerical_columns"]
        for col in num_features:
            self.full_data[col] = pd.to_numeric(self.full_data[col], errors="coerce")

        # Convert categorical features to the appropriate type
        cat_features = self.config["categorical_columns"]
        for cat_col in cat_features:
            self.full_data[cat_col] = self.full_data[cat_col].astype("category")

        train_data, test_data = train_test_split(
            self.full_data,
            test_size=0.2,
            random_state=42,
        )

        train_data, val_data = train_test_split(
            train_data,
            test_size=0.2,
            random_state=42,
        )

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def split_into_X_and_y(
        self, full_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = full_data.drop(columns=self.config["target_variable"])
        y = full_data[self.config["target_variable"]]
        return X, y

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_train, y_train = self.split_into_X_and_y(self.train_data)
        return X_train, y_train

    def get_val_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_val, y_val = self.split_into_X_and_y(self.val_data)
        return X_val, y_val

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_test, y_test = self.split_into_X_and_y(self.test_data)
        return X_test, y_test

    def save_to_catalog(
        self,
        spark: SparkSession,
    ):
        """Save the train and test sets into Databricks tables."""

        # Drop the tables if they already exist
        spark.sql(
            f"DROP TABLE IF EXISTS {self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_train_set"
        )
        spark.sql(
            f"DROP TABLE IF EXISTS {self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_val_set"
        )
        spark.sql(
            f"DROP TABLE IF EXISTS {self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_test_set"
        )

        train_set_with_timestamp = spark.createDataFrame(self.train_data).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        val_set_with_timestamp = spark.createDataFrame(self.val_data).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(self.test_data).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_train_set"
        )

        val_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_val_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_val_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.{self.config['table_name_prefix']}_test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


if __name__ == "__main__":
    hotel_dataset = HotelDataset("hotel_reservations.csv", "hotel_data_config.yaml")

    # You can save the processed datasets for later use or continue with model building
