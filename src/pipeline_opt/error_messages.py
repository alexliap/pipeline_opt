class ErrorMessages:
    EMPTY_COLUMN_LIST = "Specify the columns you want to apply the transformation to."
    EQUAL_LENGTH_LISTS = "Number of estimators must be equal to number of column lists."
    ENSEMBLE_OBJECT = (
        "The module specified is neither a VotingRegressor nor a VotingClassifier."
    )
    ENSEMBLE_LIMIT = "Max size of ensemble is greater than available estimatos. Consider removing exclusions or decreasing 'max_number_of_est'."
