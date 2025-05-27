import pandas as pd
from typing import List, Dict
from tools import load_config, Logger
from sentence_transformers import SentenceTransformer

class DataProcessor:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(__name__)
    
    def _delete_stops(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Deleting stops...')
        df = df[~df['text'].astype(str).str.startswith('#')]
        df = df.reset_index(drop=True)

        self.logger.info('Stops deleted...')
        return df
    
    def _encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Encoding labels...')
        for col in self.config['ALL_LABELS']:
            df[col] = df[col].apply(lambda x: 1 if x else 0)
        self.logger.info('Labels encoded...')
        return df

    def _embed_text(self, df: pd.DataFrame) -> pd.DataFrame:
        model = SentenceTransformer('sdadas/st-polish-paraphrase-from-distilroberta')
        
        self.logger.info('Embedding text...')
        corpus = [str(df.loc[index, 'text']) for index in df.index.to_list()]
        embeddings = model.encode(corpus)

        self.logger.info('Converting embeddings to dataframe...')
        embedding_column_names = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        
        self.logger.info('Concatenating embeddings with original dataframe...')
        df_embeddings = pd.DataFrame(embeddings, columns=embedding_column_names)
        df = pd.concat([df_embeddings, df], axis=1)

        self.logger.info('Text embedded...')
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Renaming columns to lowercase...')
        df.columns = df.columns.str.lower()
        self.logger.info('Columns renamed...')
        return df
    
    def _transform(self, name: str) -> pd.DataFrame:
        df = pd.read_csv(self.config['DATA']['DATA_PATH'] + self.config['DATA'][name])

        df = self._delete_stops(df=df)
        df = self._encode_labels(df=df)
        
        return df
        df = self._embed_text(df=df)
        df = self._rename_columns(df=df)
        
        return df
    def __call__(self, name: str) -> pd.DataFrame:
        return self._transform(name=name)
        
# if __name__ == '__main__':
#     df = DataProcessor('config.yml')('RAW_TRAIN_FILE_NAME')
#     print(df.head())
#     print(df.iloc[:, 780:])
#     print(df.columns)
#     print(df.shape)
    