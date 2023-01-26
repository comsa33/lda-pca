from gensim import models, corpora
from gensim.models import Word2Vec
from kiwipiepy import Kiwi

import functions as funcs
import preprocess as prep


prep.user_dictionary()
kiwi = Kiwi()
kiwi.load_user_dictionary('user_dictionary.txt')
kiwi.prepare()


class Word2VecModel():

    def __init__(self):
        '''주요 품사 정의'''
        self.주요품사 = ["NNG", "NNP", "VV", "VA", "XR", "SL"]
        self.용언품사 = ["VV", "VA"]

    '''형태소 분석 결과를 읽어서 주요 품사만 수집한 문서 리스트를 돌려준다.'''
    def read_documents(self, df, col):
        문서리스트 = []
        for index, row in df.iterrows():
            if row[col]:
                필터링결과 = [(token.form, token.tag) for token in row[col] if token.tag in self.주요품사]
                필터링결과 = [form+"다" if tag in self.용언품사 else form for form, tag in 필터링결과]
            문서리스트.append(필터링결과)
        return 문서리스트

    def get_w2v_model(self, df, company_name, year, col):
        df_comp = funcs.get_comp(df, company_name)
        df_comp_ = df_comp[[col, 'year']]
        df_year = df_comp_.query(f'year == {year}')
        df_year[col] = df_year[col].apply(prep.preprocess_text)
        morph_analysis = lambda x: kiwi.tokenize(x) if type(x) is str else None
        df_year['morpheme'] = df_year[col].apply(morph_analysis)
        doc_list = self.read_documents(df_year, "morpheme")
        model = Word2Vec(doc_list, window=5, min_count=3, workers=4, sg=0)
        return model
