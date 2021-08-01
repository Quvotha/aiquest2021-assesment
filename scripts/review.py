from typing import Any, Union

import pandas as pd


class Review(object):

    @staticmethod
    def cleanse_host_response_rate(host_response_rate: Any) -> Union[float, None]:
        """Convert `host_response_rate` into float if possible.

        Parameters
        ----------
        host_response_rate : Any
            `host_response_rate`

        Returns
        -------
        Union[float, None]
            Float if convertable, otherwise None.
        """
        if not isinstance(host_response_rate, str):
            return None
        try:
            response_rate = float(host_response_rate[:-1])
            return response_rate
        except TypeError:
            return None

    @staticmethod
    def extract_review_feature(
            df: pd.DataFrame, first_date_column: str = 'first_review') -> pd.DataFrame:
        """Extract feature from review information.

        Parameters
        ----------
        df : pd.DataFrame
            Must have 4 columns; `host_response_rate`, `last_review`, `number_of_reviews` and date_from_col.
            `host_response_rate` is expected to be converted into float by `cleanse_host_response_rate`.
        first_date_column : str, optional
            Column name of start date, by default 'first_review'

        Returns
        -------
        review_feature: pd.DataFrame
            Having `days_between_first_last`, `average_number_of_reviews`, `number_of_host_response`,
            and `average_number_of_host_response`.
        """
        review_feature = pd.DataFrame()
        df[first_date_column] = pd.to_datetime(df[first_date_column])
        df['last_review'] = pd.to_datetime(df['last_review'])
        review_feature['days_between_first_last'] = (df['last_review'] - df[first_date_column]) \
            .dt.days
        review_feature['average_number_of_reviews'] = (
            df['number_of_reviews'] / review_feature['days_between_first_last']
        )
        review_feature['number_of_host_response'] = df['number_of_reviews'] * df['host_response_rate']
        review_feature['average_number_of_host_response'] = (
            review_feature['number_of_host_response'] / review_feature['days_between_first_last']
        )

        return review_feature
