
    def encode_categorical_variables(self) -> pd.DataFrame:
        """
        Encode categorical variables defined in config via yaml using one-hot encoding.

        Returns:
            pd.DataFrame: DataFrame with categorical variables encoded.
        """
        categorical_column_names = self.config["categorical_columns"]

        encoded_dataframe = pd.get_dummies(
            self.full_data,
            columns=categorical_column_names,
            drop_first=True,
        )

        # Remove invalid characters from column names
        encoded_dataframe.columns = [
            re.sub(r"[ ,;{}()\n\t=]", "_", col) for col in encoded_dataframe.columns
        ]

        self.full_data = encoded_dataframe

        return self.full_data


    def normalize_numerical_features(self) -> pd.DataFrame:
        """
        Normalize/standardize the given numerical features.

        Args:
            columns_to_normalize (list[str]): List of numerical column names to be normalized.

        Returns:
            pd.DataFrame: DataFrame with normalized numerical features.
        """
        columns_to_normalize = self.config["numerical_columns"]

        scaler = StandardScaler()

        self.full_data[columns_to_normalize] = scaler.fit_transform(
            self.full_data[columns_to_normalize]
        )

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