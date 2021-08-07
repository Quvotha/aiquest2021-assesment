import logging
import os
from typing import Optional, Tuple

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import torch
import transformers
from transformers import BertTokenizer

from logging_util import timer

# 文章のベクトル化方法を示す
VECTORIZERS = (
    'bert',  # BERT
    'doc2vec',  # Doc2Vec
    'lsi_count',  # LSI (CountVectorize)
    'lsi_tfidf',  # LSI (TF-IDF)
    'lda_count',  # LDA (CountVectorize)
    'lda_tfidf'  # LDA (TF-IDF)
)


class BertSequenceVectorizer:
    # https://www.guruguru.science/competitions/16/discussions/fb792c87-6bad-445d-aa34-b4118fc378c1/
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():
            return seq_out[0][0].cpu().detach().numpy()  # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()


def make_or_load_vector(train: pd.DataFrame,
                        test: pd.DataFrame,
                        feature_dir: str,
                        logger: logging.Logger,
                        text_column: str,
                        embedder: str,
                        vector_size: Optional[int] = 200,
                        n_components: Optional[int] = 100,
                        overwrite: bool = False,
                        random_state: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if embedder not in VECTORIZERS:
        raise ValueError(f'`embedder` must be one of {VECTORIZERS} but {embedder} was given')

    # 特徴量を保存するフォルダは事前に用意しないとだめ
    if not os.path.isdir(feature_dir):
        raise ValueError(feature_dir)

    # ファイル名は勝手に決める
    # "<text_column>_<embedder>" がプレフィックス
    # サフィックスは訓練データが "_train.csv" でテストデータは "_test.csv" とする
    train_path = os.path.join(feature_dir, f'{text_column}_{embedder}_train.csv')
    test_path = os.path.join(feature_dir, f'{text_column}_{embedder}_test.csv')

    # 再作成の必要が無ければ作成済みの特徴量を読み込んで終了する
    if os.path.isfile(train_path) and os.path.isfile(test_path) and not(overwrite):
        with timer('Load vector', logger, logging.DEBUG):
            return pd.read_csv(train_path), pd.read_csv(test_path)

    with timer('Make vector', logger, logging.DEBUG):
        logger.debug('Seed is {}'.format(random_state))
        # コンペ終了1日前に色々消してしまい慌てて作り直したので 'bert' と 'doc2vec' は動くかも怪しい (#12)
        if embedder == 'bert':
            vectorizer = BertSequenceVectorizer()
            vec_train = np.concatenate(
                train[text_column].apply(vectorizer.vectorize).values
            ).reshape(train.shape[0], -1)
            vec_train = pd.DataFrame(
                data=vec_train,
                columns=[f'{embedder}vec{i + 1}' for i in range(vec_train.shape[1])]
            )
            vec_test = np.concatenate(
                test[text_column].apply(vectorizer.vectorize).values
            ).reshape(test.shape[0], -1)
            vec_test = pd.DataFrame(
                data=vec_test,
                columns=vec_train.columns.tolist()
            )
        elif embedder == 'doc2vec':
            docs_train = [text.split(' ') for text in train[text_column].values]
            tagged_docs_train = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs_train)]

            '''再現性について。
            `seed` を指定して `workers` を 1 にし、加えてモジュール実行前に環境変数 "PYTHONHASHSEED" に
            シードを指定しなければならない。アセスメントでは 1 を指定した。
            https://radimrehurek.com/gensim_3.8.3/models/doc2vec.html
            '''
            if os.environ.get("PYTHONHASHSEED") is None:
                logger.debug('`PYTHONHASHSEED` is not set')
            else:
                logger.debug('`PYTHONHASHSEED` is {}'.format(os.environ["PYTHONHASHSEED"]))
            vectorizer = Doc2Vec(tagged_docs_train, vector_size=vector_size, seed=random_state,
                                 window=2, min_count=1, workers=1)
            vec_train = np.concatenate(
                [vectorizer.infer_vector(doc) for doc in docs_train]
            ).reshape(train.shape[0], -1)
            vec_train = pd.DataFrame(
                data=vec_train,
                columns=[f'{embedder}vec{i + 1}' for i in range(vec_train.shape[1])]
            )
            docs_test = [text.split(' ') for text in test[text_column].values]
            vec_test = np.concatenate(
                [vectorizer.infer_vector(doc) for doc in docs_test]
            ).reshape(test.shape[0], -1)
            vec_test = pd.DataFrame(
                data=vec_test,
                columns=vec_train.columns.tolist()
            )
        elif embedder in ('lsi_count', 'lsi_tfidf', 'lda_count', 'lda_tfidf'):
            vectorizer_ = TfidfVectorizer() if embedder.endswith('_tfidf') else CountVectorizer()
            decomposer_ = LDA(n_components=n_components, random_state=random_state, n_jobs=-1) \
                if embedder.startswith('lda') \
                else TruncatedSVD(n_components=n_components, random_state=random_state)
            vectorizer = Pipeline(
                steps=[
                    ('vectorizer', vectorizer_),
                    ('decomposer', decomposer_)
                ]
            )
            vec_train = vectorizer.fit_transform(train[text_column])
            vec_train = pd.DataFrame(
                data=vec_train,
                columns=[f'{embedder}vec{i + 1}' for i in range(n_components)]
            )
            vec_test = vectorizer.transform(test[text_column])
            vec_test = pd.DataFrame(
                data=vec_test,
                columns=vec_train.columns.tolist()
            )
        else:
            raise NotImplementedError

    with timer('Save vector', logger, logging.DEBUG):
        vec_train.to_csv(train_path, index=False)
        vec_test.to_csv(test_path, index=False)

    return vec_train, vec_test
