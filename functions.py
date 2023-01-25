import warnings
import re
from itertools import chain
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from kiwipiepy import Kiwi

import mongodb


warnings.filterwarnings(action='ignore')

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothic'


client = mongodb.client
db_names = mongodb.db_names

col_reordered =  [
    '_id',
    'Review_Id',
    'Location',
    'DatePost',
    'Department',
    'Employee_status',
    'Company_name',
    'Title',
    'Pros',
    'Cons',
    'To_Management',
    'Ratings',
    'Culture',
    'WorkLifeBalance',
    'Benefits',
    'Management',
    'Opportunity',
    'Potential',
    'Recommend'
]

def get_collections(db_no):
    db = client.get_database(db_names[db_no])
    coll_names = {}
    for i, coll in enumerate(db.list_collection_names()):
        coll_names[i] = coll
    return coll_names

def get_df(coll, collection_no):
    cursor = coll.find()
    df = pd.DataFrame(list(cursor))
    return df[col_reordered]

def get_comp(df, company_name):
    df_ = df[df['Company_name'] == company_name]
    df_['DatePost'] = pd.to_datetime(df_['DatePost'], errors='coerce')
    df_['year'] = df_['DatePost'].apply(lambda x: x.year)
    return df_

def get_value_counts(df, year, field):
    df_ = df.query(f'year == {year}')
    cnt = df_[field].value_counts().values.tolist()
    return (cnt[0]/sum(cnt))*100

def get_fluctuation2(df, field):
    years = []
    trends = []
    for year in range(df.year.min(), df.year.max()+1):
        years.append(year)
        trends.append(get_value_counts(df, year, field))
    return years, trends

def get_mean(df, year, field):
    df_ = df.query(f'year == {year}')
    return df_[field].mean()

def get_fluctuation(df, field):
    years = []
    trends = []
    for year in range(df.year.min(), df.year.max()):
        years.append(year)
        trends.append(get_mean(df, year, field))
    return years, trends

def draw_fluctuation(df, company_name):
    df_comp = get_comp(df, company_name)
    fields = ['Ratings', 'Culture', 'WorkLifeBalance', 'Benefits', 'Management', 'Opportunity']
    fields2 = ['Potential', 'Recommend']

    for field in fields:
        years, trends = get_fluctuation(df_comp, field)
        plt.figure(figsize=(7, 2))
        fig1 = sns.barplot(x=years, y=trends, palette='crest')
        plt.title(f'[{field}] annual trends')

    for field in fields2:
        years, trends = get_fluctuation2(df_comp, field)
        plt.figure(figsize=(7, 2))
        fig2 = sns.barplot(x=years, y=trends, palette='flare')
        plt.title(f'[{field}] annual trends')
    return fig1, fig2

main_words = ["NNG", "NNP", "VV", "VA", "XR"]
sub_words = ["VV", "VA"]

def filter_docs(df, col):
    result = []
    for index, row in df.iterrows(): 
        if row[col]:
            filetered = [(token.form, token.tag) for token in row[col] if token.tag in main_words]
            filetered = [form+"다" if tag in sub_words else form for form, tag in filetered]
        result.append(filetered)
    return result

def get_most_common(df, col, n:int):
    kiwi = Kiwi()
    # kiwi.load_user_dictionary('user_dictionary.txt')
    kiwi.prepare()
    
    compile = re.compile("\W+")
    df['text'] = df[col].apply(lambda x: compile.sub(" ", x))

    morph_analysis = lambda x: kiwi.tokenize(x) if type(x) is str else None
    df['text'] = df['text'].apply(morph_analysis)

    result = filter_docs(df, 'text')
    flatten = list(chain(*result))
    counter = Counter(flatten)
    return counter.most_common(n)

def get_most_common_by_year(df_comp, year, col, n:int):
    df_year = df_comp.query(f'year == {year}')
    data = get_most_common(df_year, col, n)
    return pd.DataFrame(data, columns=['words', year]).set_index('words')

def get_all_most_common_join_df(df, company_name, col, n):
    df_comp = get_comp(df, company_name)
    df_join = pd.DataFrame(get_most_common(df_comp, col, n), columns=['words', 'total']).set_index('words')
    for year in range(df_comp.year.min(), df_comp.year.max()+1):
        df_ = get_most_common_by_year(df_comp, year, col, n)
        df_join = df_join.join(df_, how='outer')
    return df_join

def draw_heatmap_most_common(df, company_name, col, n):
    df_join = get_all_most_common_join_df(df, company_name, col, n)
    df_join.fillna(0, inplace=True)
    plt.figure(figsize=(15,15))
    ax = sns.heatmap(df_join.iloc[:,1:], annot=True, fmt=".0f")
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.title(f"{company_name} {col} TOP WORDS {n}")
    plt.show()
    
