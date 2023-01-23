from gensim import models, corpora
from gensim.models import Word2Vec
from kiwipiepy import Kiwi

import functions as funcs


kiwi = Kiwi()
# kiwi.load_user_dictionary('user_dictionary.txt')
kiwi.prepare()


class LDA_Model():

    def __init__(self):
        '''주요 품사 정의'''
        self.주요품사 = ["NNG", "NNP", "VV", "VA", "XR"]
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

    '''주어진 문서 집합으로 문서-어휘 행렬을 만들어 돌려준다.'''
    def build_doc_term_mat(self, 문서리스트):
        dictionary = corpora.Dictionary(문서리스트)
        corpus = [dictionary.doc2bow(문서) for 문서 in 문서리스트]
        return corpus, dictionary

    '''문서-어휘 행렬을 TF-IDF 문서-단어 행렬로 변환한다.'''
    def build_corpus_tfidf(self, corpus):
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    def get_lda_model(self, df, company_name, year, col):
        df_comp = funcs.get_comp(df, company_name)
        df_comp_ = df_comp[[col, 'year']]
        df_year = df_comp_.query(f'year == {year}')
        morph_analysis = lambda x: kiwi.tokenize(x) if type(x) is str else None
        df_year['morpheme'] = df_year[col].apply(morph_analysis)
        doc_list = self.read_documents(df_year, "morpheme")
        print(f"{year} DATA LENGTH :", len(doc_list))
        model = Word2Vec(doc_list, window=5, min_count=3, workers=4, sg=0)
        return model
