from pyspark.sql import SparkSession

from sarah_hotel_reservation_data.data_processing import HotelDataset

hotel_data = HotelDataset(
    data_filepath="../data/Hotel Reservations.csv",
    yaml_file_path="../hotel_project_config.yaml",
)
hotel_data.run_data_preparation()

spark = SparkSession.builder.getOrCreate()

hotel_data.save_to_catalog(spark=spark)
