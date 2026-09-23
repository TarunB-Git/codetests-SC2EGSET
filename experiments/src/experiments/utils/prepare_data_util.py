import pandas as pd
from sklearn.preprocessing import StandardScaler


def prep_for_dim_reduction(grouped_field: str, result_df: pd.DataFrame):
    """
    Helper function that prepares data for final experiment stage of dimensionality reduction.

    :param grouped_field: Specifies the name of a field that\
    was used for the groupby operation.
    :type grouped_field: str
    :param result_df: Specifies a dataframe that will be\
    standardized and used for dimensionality reduction.
    :type result_df: pd.DataFrame
    :return: Returns data prepared for the dimensionality reduction task.
    """

    unique_grouped = result_df[grouped_field].unique()

    # Mapping target to digits:
    unique_map_learning = {
        unique_field: str(i) for i, unique_field in enumerate(unique_grouped)
    }
    unique_map_display = {v: k for k, v in unique_map_learning.items()}
    result_df[grouped_field] = result_df[grouped_field].map(unique_map_learning)

    # Standardizing the data:
    without_grouped_field = result_df.loc[:, result_df.columns != grouped_field]
    standardized_data = StandardScaler().fit_transform(without_grouped_field)

    return standardized_data, unique_map_display
