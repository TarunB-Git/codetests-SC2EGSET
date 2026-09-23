from typing import Dict, List

import pandas as pd


def drop_unused_fields(
    grouped_dataframes: Dict[str, pd.DataFrame],
    fields_to_drop: List[str],
    unique_games: int,
) -> Dict[str, pd.DataFrame]:
    """
    Helper function that drops unused fields in data preparation\
    step for dimensionality reduction.

    :param grouped_dataframes: Specifies a dataframe that will be changed.
    :type grouped_dataframes: Dict[str, pd.DataFrame]
    :param fields_to_drop: Specifies which fields of the dataframe should be dropped.
    :type fields_to_drop: List[str]
    :param unique_games: Specifies number of unique games for logging purposes.
    :type unique_games: int
    :return: Returns a dictionary that maps the dataframe containing\
    the transformed data to a string value signifying which field was using\
    for the groupby operation previously.
    :rtype: Dict[str, pd.DataFrame]
    """

    # Dropping game time from outcome as it surely does not differentiate players:
    dropped_outcome = grouped_dataframes["outcome"].drop(labels=fields_to_drop, axis=1)
    grouped_dataframes["outcome"] = dropped_outcome

    # Dropping game hash:
    for grouped_field, result_df in grouped_dataframes.items():
        # print(f"Grouped unique games by {grouped_field}: {grouped_unique_games}")
        # print(
        #     f"Unique games grouped by {grouped_field} and grouped games match: {unique_games == grouped_unique_games}"
        # )
        dropped_hash = result_df.drop(labels=["game_hash"], axis=1)
        grouped_dataframes[grouped_field] = dropped_hash

    return grouped_dataframes
