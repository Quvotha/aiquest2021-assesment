import pandas as pd


class SimilarHomeSearcher(object):

    def __init__(self, df: pd.DataFrame):
        """Initializer.

        Parameters
        ----------
        df : pd.DataFrame
            民泊の情報。`description`, `city`, `zipcode5`, `name`, `y` のカラムが存在しなければならない。
            データ型は `y` は float でその他は文字列であることが期待される。
        """
        self.search_space1 = df \
            .dropna() \
            .groupby(['description', 'city', 'zipcode5', 'name'])['y'] \
            .mean() \
            .sort_index()

        self.search_space2 = df \
            .dropna() \
            .groupby(['description', 'city', 'zipcode5'])['y'] \
            .mean() \
            .sort_index()

        self.search_space3 = df \
            .dropna() \
            .groupby(['description', 'city'])['y'] \
            .mean() \
            .sort_index()

    def get_y_of_similar(
            self, description: str, city: str, zipcode5: str, name: str, default: float) -> float:
        """同じような民泊を探して見つかればその `y` の平均を返す。

        以下の条件群を数字順に検査し、最初に該当した条件に従って結果を返す。
        条件１）description, city, zipcode5, name が一致する民泊が存在する場合：それらの `y` の平均値を返す
        条件２）description, city, zipcode5 が一致する民泊が存在する場合：それらの `y` の平均値を返す
        条件３）description, city が一致する民泊が存在する場合：それらの `y` の平均値を返す
        条件４）上記のどの条件にも該当しない場合：default を返す

        Parameters
        ----------
        description : str
            `description` の検索条件。。
        city : str
            `city` の検索条件。
        zipcode5 : str
            `zipcode5` の検索条件。
        name : str
            `name` の検索条件。
        default : float
            条件４に該当した場合は `default` の値を返す。

        Returns
        -------
        average `y` or default: float
            同じような民泊の `y` の平均値もしくは `default`.
        """
        try:
            y = self.search_space1.loc[(description, city, zipcode5, name)]
            return y.mean() if isinstance(y, pd.Series) else y
        except KeyError:
            pass

        try:
            y = self.search_space2.loc[(description, city, zipcode5)]
            return y.mean() if isinstance(y, pd.Series) else y
        except KeyError:
            pass

        try:
            y = self.search_space3.loc[(description, city)]
            return y.mean() if isinstance(y, pd.Series) else y
        except KeyError:
            return default
