import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(filepath):
    """
    Load the dataset from the given file path.

    Args:
        filepath (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset (if any).

    Args:
        df (pd.DataFrame): DataFrame with possible missing values.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Example: Impute missing values (you can change the strategy as needed)
    df.fillna(df.median(), inplace=True)
    return df


def encode_categorical_variables(
    dataframe: pd.DataFrame, categorical_column_names: list[str]
) -> pd.DataFrame:
    """
    Encode categorical variables using one-hot encoding.

    Args:
        dataframe (pd.DataFrame): Original DataFrame.
        categorical_column_names (list): List of categorical columns to be encoded.

    Returns:
        pd.DataFrame: DataFrame with categorical variables encoded.
    """
    encoded_dataframe = pd.get_dummies(
        dataframe,
        columns=categorical_column_names,
        drop_first=True,
    )

    encoded_dataframe.drop(columns=categorical_column_names, inplace=True)

    return encoded_dataframe


def normalize_numerical_features(
    dataframe: pd.DataFrame, columns_to_normalize: list[str]
) -> pd.DataFrame:
    """
    Normalize/standardize the given numerical features.

    Args:
        dataframe (pd.DataFrame): DataFrame with numerical features.
        columns_to_normalize (list[str]): List of numerical column names to be normalized.

    Returns:
        pd.DataFrame: DataFrame with normalized numerical features.
    """
    scaler = StandardScaler()

    dataframe[columns_to_normalize] = scaler.fit_transform(
        dataframe[columns_to_normalize]
    )

    return dataframe


def convert_target_variable(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the target variable 'booking_status' into binary.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'booking_status' column.

    Returns:
        pd.DataFrame: DataFrame with 'booking_status' as a binary column.
    """
    binary_target = dataframe["booking_status"].apply(
        lambda status: 1 if status == "Canceled" else 0
    )
    dataframe["booking_status"] = binary_target
    return dataframe


def split_dataframe_into_train_test_sets(
    dataframe: pd.DataFrame,
    target_column_name: str,
    test_set_proportion: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the given DataFrame into training and test sets.

    Args:
        dataframe (pd.DataFrame): DataFrame to be split.
        target_column_name (str): Name of the target column.
        test_set_proportion (float): Proportion of the data to be used as the test set.
        random_seed (int): Seed for random splitting.

    Returns:
        tuple: X_train, X_test, y_train, y_test sets.
    """
    features = dataframe.drop(columns=[target_column_name])
    target = dataframe[target_column_name]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_set_proportion, random_state=random_seed
    )

    return X_train, X_test, y_train, y_test


# Example: Main function to run the data preparation steps
def data_preparation_pipeline(filepath):
    """
    Pipeline function to run all data preparation steps sequentially.

    Args:
        filepath (str): Path to the dataset.

    Returns:
        tuple: X_train, X_test, y_train, y_test prepared for training.
    """
    # Load the dataset
    dataframe = load_dataset(filepath)

    # Handle missing values
    dataframe = handle_missing_values(dataframe)

    # Encode categorical variables
    categorical_columns = [
        "column1",
        "column2",
    ]  # Placeholder for actual categorical columns
    dataframe = encode_categorical_variables(dataframe, categorical_columns)

    # Normalize numerical features
    numerical_columns = ["lead_time", "avg_price_per_room"]
    dataframe = normalize_numerical_features(dataframe, numerical_columns)

    # Convert target variable
    dataframe = convert_target_variable(dataframe)

    # Split the data
    X_train, X_test, y_train, y_test = split_dataframe_into_train_test_sets(
        dataframe, target_column_name="booking_status"
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_preparation_pipeline(
        "hotel_reservations.csv"
    )

    # You can save the processed datasets for later use or continue with model building
