# aiquest2021-assesment

|RMSE|RANK|
|:---:|:---:|
|147.7703573|147 of 964|

# Solution

コードは全て Python (3.7.0). コードは [05-07.ipynb](experiments/05-07.ipynb) を参照。

## Preprocessing
前処理の大半は [feature_engineering.py](scripts/feature_engineering.py) で行われる。

- `amenities`  
  半角カンマ区切りで列挙されているアメニティを全て抽出し train.csv と test.csv の両方に登場するものを対象に One-Hot encoding し PCA で次元削減を行った。
- `neighborfood`  
  欠損値には専用のカテゴリを割り当てた。
- `cleaning_fee`, `host_has_profile_pic`, `host_identity_verified`, `instant_bookable`  
  二値のフラグ項目として利用。欠損値は False として扱った。
- `host_response_rate`  
  文字列→実数に変換。
- `first_review`, `last_review`  
  年月を抽出し特徴量に追加。日数差を特徴量に追加。日数差を `number_of_review` で除算して1日の平均レビュー数として特徴量に追加。
- `zipcode`  
  米国の郵便番号の体系にあうようにやれる範囲で正規化した上で先頭から1桁目、5桁目を特徴量とした。欠損値は `city` である程度類推した。
- `thumbnail_url`  
  サムネイル有無をフラグ項目として追加。
- `name`, `description`  
  両者を結合しベクトル化した (CountVectorizer -> LDA).
- `latitude`, `longitude`, , `cancellation_policy`, `bed_type`, `city`, `neighbourhood`, `property_type`, `room_type`, `id` 以外の連続変数  
  特徴量として利用。

## Model
`y` > 200 の民泊価格を過小評価する傾向が著しかったため以下の工夫を実施したがあまり改善できなかった。

- `y` > 200 かどうかを予測する二値分類器を CatBoost で訓練（分類器）
- `y` > 200 のデータだけを使い LightGBM で `y` の予測モデルを訓練（回帰器①）
- `y` <=> 200 のデータだけを使い LightGBM で `y` の予測モデルを訓練（回帰器②）
- 分類器の予測結果が `y` > 200 なら回帰器①の予測結果を、そうでなければ回帰器②の予測結果を採用

なお回帰器の訓練では `y` を対数変換している。

## Cross validation
`y` を3層に binning し bin を層として 5-Fold stratified cross validation を行った。

# Environment

## Hardware

全処理をPCで実施。スペックは以下の通り。

- OS: Microsoft Windows 10 Home
- RAM: 16GB
- CPU: Intel(R) Core(TM) i5-10300HCPU @ 2.5GHz (4-cores)
- No GPU

## Virtual environment
Miniconda で管理。使用したパッケージは [conda_package_list.txt](conda_package_list.txt).