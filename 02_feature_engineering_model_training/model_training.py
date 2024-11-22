import mlflow

from sarah_hotel_reservation_data.data_processing import HotelDataset

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Shared/hotel-data-sarah")
mlflow.set_experiment_tags({"repository_name": "hotel-data-sarah"})


hotel_data = HotelDataset(
    data_filepath="../data/Hotel Reservations.csv",
    yaml_file_path="../hotel_project_config.yaml",
)
hotel_data.run_data_preparation()
X_train, y_train = hotel_data.get_train_data()
X_val, y_val = hotel_data.get_val_data()

X_train.head()
