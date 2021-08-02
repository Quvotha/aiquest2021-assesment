from itertools import chain
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import pandas as pd

from amenities import Amenities
from logging_util import timer
from review import Review
from zipcode import ZipCodePreprocessor

FLAG_COLUMNS = ['cleaning_fee', 'host_has_profile_pic',
                'host_identity_verified', 'instant_bookable']
DATE_COLUMNS = ['first_review', 'host_since', 'last_review']
NEIGHTBOURHOOD_BLANK_CATEGORY = 'No data'


def make_or_load_features(train: pd.DataFrame, test: pd.DataFrame,
                          train_path: str, test_path: str,
                          logger: logging.Logger,
                          overwrite: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 再作成の必要が無ければ作成済みの特徴量を読み込んで終了する
    if os.path.isfile(train_path) and os.path.isfile(test_path) and not(overwrite):
        with timer('Load features', logger, logging.DEBUG):
            return pd.read_csv(train_path), pd.read_csv(test_path)

    # 特徴量の出力先フォルダはあらかじめ作っておかないとエラー終了する
    if not os.path.isdir(os.path.dirname(train_path)):
        raise ValueError(train_path)
    if not os.path.isdir(os.path.dirname(test_path)):
        raise ValueError(test_path)

    # 特徴量を作成する
    with timer('Make feature', logger, logging.DEBUG):

        '''フラグ項目を int 型にして欠損は一律 0 埋めする'''
        flag_map = {
            't': 1,
            'f': 0
        }
        for flag_column in FLAG_COLUMNS:
            train[flag_column] = train[flag_column].map(flag_map).fillna(0).astype('int')
            test[flag_column] = test[flag_column].map(flag_map).fillna(0).astype('int')

        '''近隣情報はブランクが多いので専用のカテゴリを割り当てる'''
        assert(NEIGHTBOURHOOD_BLANK_CATEGORY not in train['neighbourhood'])
        assert(NEIGHTBOURHOOD_BLANK_CATEGORY not in test['neighbourhood'])
        train['neighbourhood'].fillna(NEIGHTBOURHOOD_BLANK_CATEGORY, inplace=True)
        test['neighbourhood'].fillna(NEIGHTBOURHOOD_BLANK_CATEGORY, inplace=True)

        '''amenities から特徴を抽出する
        1. 訓練データとテストデータの両方に登場するアメニティを抽出する
        2. 1. で抽出したアメニティを対象に訓練データに対してアメニティの有無を one-hot 表現に置換する
        3. 2. と同様の処理をテストデータに対しておこなう
        '''
        # 1. 訓練データとテストデータの両方に登場するアメニティを抽出する
        amenity_lists_train = train['amenities'].apply(Amenities.tolist).tolist()
        amenity_lists_test = test['amenities'].apply(Amenities.tolist).tolist()
        amenity_names_train = set(chain.from_iterable(amenity_lists_train))
        amenity_names_test = set(chain.from_iterable(amenity_lists_test))
        amenity_names = tuple(amenity_names_train & amenity_names_test)
        # 2. 1. で抽出したアメニティを対象に訓練データに対してアメニティの有無を one-hot 表現に置換する
        amenity_onehot_feature_names = [f'has_{a}_amenity' for a in amenity_names]
        onehot_vec_train = []
        for amenity_list in amenity_lists_train:
            onehot_vec_ = [0] * len(amenity_names)
            for i, amenity_name in enumerate(amenity_names):
                if amenity_name in amenity_list:
                    onehot_vec_[i] = 1
            onehot_vec_train.append(onehot_vec_)
        onehot_vec_train = pd.DataFrame(onehot_vec_train)
        onehot_vec_train.columns = amenity_onehot_feature_names
        onehot_vec_train.index = train.index
        train = pd.concat([train, onehot_vec_train], axis=1)
        train.drop(columns=['amenities'], inplace=True)
        # 3. 2. と同様の処理をテストデータに対しておこなう
        onehot_vec_test = []
        for amenity_list in amenity_lists_test:
            onehot_vec_ = [0] * len(amenity_names)
            for i, amenity_name in enumerate(amenity_names):
                if amenity_name in amenity_list:
                    onehot_vec_[i] = 1
            onehot_vec_test.append(onehot_vec_)
        onehot_vec_test = pd.DataFrame(onehot_vec_test)
        onehot_vec_test.columns = amenity_onehot_feature_names
        onehot_vec_test.index = test.index
        test = pd.concat([test, onehot_vec_test], axis=1)
        test.drop(columns=['amenities'], inplace=True)

        '''レビューの特徴を抽出する'''
        # 訓練データに対して
        train['host_response_rate'] = np.vectorize(
            Review.cleanse_host_response_rate)(
            train['host_response_rate'])
        review_feature_train = Review.extract_review_feature(
            train[['host_response_rate', 'last_review', 'number_of_reviews', 'first_review']]
        )
        review_feature_train.index = train.index
        train = pd.concat([train, review_feature_train], axis=1)
        # テストデータに対して
        test['host_response_rate'] = np.vectorize(
            Review.cleanse_host_response_rate)(
            test['host_response_rate'])
        review_feature_test = Review.extract_review_feature(
            test[['host_response_rate', 'last_review', 'number_of_reviews', 'first_review']]
        )
        review_feature_test.index = test.index
        test = pd.concat([test, review_feature_test], axis=1)

        '''所在地に関連する特徴を抽出する'''
        # 訓練データに対して
        zipcode_preprocessor = ZipCodePreprocessor().fit(train)
        zip_features_train = zipcode_preprocessor.transform(train)
        zip_features_train.index = train.index
        train = pd.concat([train, zip_features_train], axis=1)
        train.drop(columns=['zipcode'], inplace=True)
        # テストデータに対して
        zip_features_test = zipcode_preprocessor.transform(test)
        zip_features_test.index = test.index
        test = pd.concat([test, zip_features_test], axis=1)
        test.drop(columns=['zipcode'], inplace=True)

        '''サムネイル画像の有無を特徴にする'''
        # 訓練データに対して
        train['has_thumbnail'] = train['thumbnail_url'].isnull() * 1
        train.drop(columns=['thumbnail_url'], inplace=True)
        # テストデータに対して
        test['has_thumbnail'] = test['thumbnail_url'].isnull() * 1
        test.drop(columns=['thumbnail_url'], inplace=True)

        '''日付の項目は year, month の情報だけ残す'''
        for date_column in DATE_COLUMNS:
            train[date_column] = pd.to_datetime(train[date_column])
            train[f'{date_column}_year'] = train[date_column].dt.year
            train[f'{date_column}_month'] = train[date_column].dt.month
            test[date_column] = pd.to_datetime(test[date_column])
            test[f'{date_column}_year'] = test[date_column].dt.year
            test[f'{date_column}_month'] = test[date_column].dt.month
        train.drop(columns=DATE_COLUMNS, inplace=True)
        test.drop(columns=DATE_COLUMNS, inplace=True)

        '''訓練データとテストデータの片方にしか登場しないカテゴリは専用の値に置換する
        数値への変換も行う
        '''
        for c in train.select_dtypes('object').columns:
            replacement = f'{c}_unshared'  # 置換する値
            train_set = set(train[c].tolist())
            test_set = set(test[c].tolist())
            shared_set = train_set & test_set
            train_only = list(train_set - shared_set)
            if train_only:
                # 訓練データにしか登場しないカテゴリを置換する
                assert(replacement not in train[c].tolist())
                train[c].replace(train_only, replacement, inplace=True)
            test_only = list(test_set - shared_set)
            if test_only:
                # テストデータにしか登場しないカテゴリを置換する
                assert(replacement not in test[c].tolist())
                test[c].replace(test_only, replacement, inplace=True)
            map_object = {v: i for i, v in enumerate(train[c].unique())}
            train[c] = train[c].map(map_object)
            test[c] = test[c].map(map_object)

        '''値が1つしかない特徴を除外する'''
        columns_only_1_value = [c for c in train.columns if train[c].nunique(dropna=False) == 1]

        if columns_only_1_value:
            train.drop(columns=columns_only_1_value, inplace=True)
            test.drop(columns=columns_only_1_value, inplace=True)
            logger.debug('Drop {} features; ({})'
                         .format(len(columns_only_1_value), ' '.join(columns_only_1_value)))

        '''name, description は除外する'''
        train.drop(columns=['name', 'description'], inplace=True)
        test.drop(columns=['name', 'description'], inplace=True)

        '''ターゲットを除外する'''
        train.drop(columns=['y'], inplace=True)

    # 特徴量を保存する
    with timer('Save features', logger, logging.DEBUG):
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
    return train, test
