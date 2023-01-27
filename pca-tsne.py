import numpy as np
import pickle
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pyLDAvis.gensim_models
import plotly.graph_objs as go
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import functions as funcs
from word2vec import Word2VecModel
from lda import LDA_Model
import mongodb


mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothic'


st.set_page_config(
    page_title="리뷰데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.experimental_memo
def get_df():
    client = mongodb.client
    db_names = mongodb.db_names
    db = client.get_database(db_names[1])
    coll_names = funcs.get_collections(1)
    coll = db[coll_names[5]]

    df = funcs.get_df(coll, 5)

    filename = ['jp_comp_name_list']
    comp_name_ls = tuple(pickle.load(open(filename[0], 'rb')))
    return df, comp_name_ls


def append_list(sim_words, words):
    list_of_words = []
    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
    return list_of_words


def display_scatterplot_3D(
    model,
    user_input=None,
    words=None, label=None,
    color_map=None,
    annotation='On',
    dim_red='PCA',
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
    sample=10
):
    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.wv.index_to_key), sample)
        else:
            words = [word for word in model.wv.index_to_key]

    word_vectors = np.array([model.wv[w] for w in words])

    if dim_red == 'PCA':
        three_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :3]
    else:
        three_dim = TSNE(n_components=3, random_state=0, perplexity=perplexity, learning_rate=learning_rate, n_iter=iteration).fit_transform(word_vectors)[:, :3]

    color = 'blue'
    quiver = go.Cone(
                x=[0, 0, 0],
                y=[0, 0, 0],
                z=[0, 0, 0],
                u=[1.5, 0, 0],
                v=[0, 1.5, 0],
                w=[0, 0, 1.5],
                anchor="tail",
                colorscale=[[0, color], [1, color]],
                showscale=False
            )
    data = [quiver]

    count = 0
    for i in range(len(user_input)):
        trace = go.Scatter3d(
                    y=three_dim[count:count+topn, 1],
                    x=three_dim[count:count+topn, 0],
                    z=three_dim[count:count+topn, 2],
                    text=words[count:count+topn] if annotation == 'On' else '',
                    name=user_input[i],
                    textposition="top center",
                    textfont_size=15,
                    mode='markers+text',
                    marker={
                        'size': 8,
                        'opacity': 0.8,
                        'color': 2
                    }
                )
        data.append(trace)
        count += topn

    trace_input = go.Scatter3d(
                    x=three_dim[count:, 0],
                    y=three_dim[count:, 1],
                    z=three_dim[count:, 2],
                    text=words[count:],
                    name='input words',
                    textposition="top center",
                    textfont_size=15,
                    mode='markers+text',
                    marker={
                        'size': 8,
                        'opacity': 1,
                        'color': 'grey'
                    }
                )
    data.append(trace_input)

# Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="grey"
            )
        ),
        font=dict(
            family="Courier New",
            size=15
        ),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    st.plotly_chart(plot_figure)


def horizontal_bar(word, similarity):
    similarity = [round(elem, 2) for elem in similarity]
    data = go.Bar(
                x=similarity,
                y=word,
                orientation='h',
                text=similarity,
                marker_color=4,
                textposition='auto'
            )
    layout = go.Layout(
                font=dict(size=20),
                xaxis=dict(showticklabels=False, automargin=True),
                yaxis=dict(showticklabels=True, automargin=True, autorange="reversed"),
                margin=dict(t=20, b=20, r=10)
            )
    plot_figure = go.Figure(data=data, layout=layout)
    st.plotly_chart(plot_figure)


def display_scatterplot_2D(
    model,
    user_input=None,
    words=None,
    label=None,
    color_map=None,
    annotation='On',
    dim_red='PCA',
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
    sample=10
):

    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.wv.index_to_key), sample)
        else:
            words = [word for word in model.wv.index_to_key]

    word_vectors = np.array([model.wv[w] for w in words])

    if dim_red == 'PCA':
        two_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :2]
    else:
        two_dim = TSNE(random_state=0, perplexity=perplexity, learning_rate=learning_rate, n_iter=iteration).fit_transform(word_vectors)[:, :2]

    data = []
    count = 0
    for i in range(len(user_input)):
        trace = go.Scatter(
            x=two_dim[count:count+topn, 0],
            y=two_dim[count:count+topn, 1],
            text=words[count:count+topn] if annotation == 'On' else '',
            name=user_input[i],
            textposition="top center",
            textfont_size=15,
            mode='markers+text',
            marker={
                'size': 8,
                'opacity': 0.8,
                'color': 2
            }
        )
        data.append(trace)
        count += topn

    trace_input = go.Scatter(
                    x=two_dim[count:, 0],
                    y=two_dim[count:, 1],
                    text=words[count:],
                    name='input words',
                    textposition="top center",
                    textfont_size=15,
                    mode='markers+text',
                    marker={
                        'size': 8,
                        'opacity': 1,
                        'color': 'grey'
                    }
                )
    data.append(trace_input)

# Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white",
            font_size=20,
            font_family="Courier New"
        ),
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="grey"
            )
        ),
        font=dict(
            family=" Courier New ",
            size=15
        ),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    st.plotly_chart(plot_figure)


df, comp_name_ls = get_df()
col_dic = {'장점': 'Pros', '단점': 'Cons', '경영진에게': 'To_Management'}


@st.experimental_memo
def get_model(df, company_name, year, col):
    w2v = Word2VecModel()
    model = w2v.get_w2v_model(df, company_name, year, col)
    return model

with st.sidebar:
    st.text('---[데이터 필터]---')
year = st.sidebar.slider(
    '연도를 선택하세요.',
    2014, 2022, (2021)
)
col = st.sidebar.selectbox(
    "분석 텍스트 필드를 선택하세요.",
    ('장점', '단점', '경영진에게')
)
company_name = st.sidebar.selectbox(
    "회사명을 입력/선택하세요.",
    comp_name_ls
)

st.title('[그레이비랩 기업부설 연구소 / AI lab.]')

tab1, tab2, tab3, tab4 = st.tabs(["PCA & TSNE 분석", "LDA 분석", "HEATMAP 분석", "평점 트렌드 분석"])

with tab1:
    st.subheader('PCA & STNE 분석 시각화를 위한 웹앱입니다.')
    with st.expander("가이드 펼쳐보기"):
        st.markdown(
            """
>  1. 먼저 보고 싶은 시각화 분석 차원 축소 기법과 차원을 선택합니다. 차원 축소 기법은 PCA와 TSNE가 있습니다. 차원은 2D와 3D의 두 가지 옵션이 있습니다.
>  2. 다음으로 사이드바에서 분석할 연도, 회사, 필드('장점', '단점', '경영진에게')를 선택하세요.
>  3. 그 다음, 분석할 단어를 입력합니다. 한 단어와 다른 단어를 쉼표(,)로 구분하여 두 개 이상의 단어를 입력할 수 있습니다.
>  4. 사이드바의 슬라이더를 사용하여 시각화하려는 입력 단어와 관련된 단어의 양을 선택할 수 있습니다. 이는 임베딩 공간에서 단어 벡터 간의 코사인 유사성을 계산하여 수행됩니다.
>  5. 마지막으로 시각화에서 텍스트 주석을 활성화하거나 비활성화하는 옵션이 있습니다.
"""
        )
    
    model = get_model(df, company_name, year, col_dic[col])

    col1_tab1, col2_tab1 = st.columns([1, 4])
    with col1_tab1:
        dim_red = st.selectbox(
            '⁜ 차원 축소 기법을 선택하세요.',
            ('PCA', 'TSNE')
        )
        dimension = st.selectbox(
            "⁜ 시각화 차원을 선택하세요.",
            ('2D', '3D')
            )
        user_input = st.text_input(
            "⁜ 분석하고자 하는 단어를 입력하세요. 한 개 이상의 단어를 입력하려면 콤마(,)로 단어를 분리하세요.",
            ''
        )
        top_n = st.slider(
            '⁜ 입력 단어와 관련된 시각화에 보여줄 단어의 수를 선택하세요.',
            5, 200, (15)
        )
        sample_n = st.slider(
            '⁜ 어떠한 단어도 입력하지 않았다면, 보여줄 샘플 단어의 수를 선택하세요.',
            5, 200, (15)
        )
        annotation = st.radio(
            "⁜ 주석을 키고 끌 수 있습니다.",
            ('On', 'Off')
        )
    if dim_red == 'TSNE':
        perplexity = st.slider(
            '⁜ TSNE모델 훈련을 위한 perplexity을 조정하십시오. perplexity는 매니폴드 학습 알고리즘에서 사용되는 가장 가까운 이웃의 수와 관련이 있습니다. 더 큰 데이터 세트에는 일반적으로 더 큰 perplexity이 필요합니다.',
            0, 50, (5)
        )

        learning_rate = st.slider(
            '⁜ TSNE모델 훈련을 위한 learning rate를 조정하십시오.',
            10, 1000, (200)
        )

        iteration = st.slider(
            '⁜ TSNE모델 훈련을 위한 iteration 수를 조정하십시오',
            250, 100000, (1000)
        )
    else:
        perplexity = 0
        learning_rate = 0
        iteration = 0

    if user_input == '':
        similar_word = None
        labels = None
        color_map = None
    else:
        user_input = [x.strip() for x in user_input.split(',')]
        result_word = []
        for words in user_input:
            sim_words = model.wv.most_similar(words, topn=top_n)
            sim_words = append_list(sim_words, words)
            result_word.extend(sim_words)
        similar_word = [word[0] for word in result_word]
        similarity = [word[1] for word in result_word]
        similar_word.extend(user_input)
        labels = [word[2] for word in result_word]
        label_dict = dict([(y, x+1) for x, y in enumerate(set(labels))])
        color_map = [label_dict[x] for x in labels]

    with col2_tab1:
        if dimension == '2D':
            st.subheader('2D 시각화')
            st.markdown('> 각 포인트에 대한 자세한 내용을 보려면(주석을 읽기 어려운 경우를 대비하여) 각 포인트 주변을 마우스로 가리키면 단어를 볼 수 있습니다. 시각화의 오른쪽 상단 모서리에 있는 확장 기호를 클릭하여 시각화를 확장할 수 있습니다.')
            display_scatterplot_2D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n, sample_n)
        else:
            st.subheader('3D 시각화')
            st.markdown('> 각 포인트에 대한 자세한 내용을 보려면(주석을 읽기 어려운 경우를 대비하여) 각 포인트 주변을 마우스로 가리키면 단어를 볼 수 있습니다. 시각화의 오른쪽 상단 모서리에 있는 확장 기호를 클릭하여 시각화를 확장할 수 있습니다.')
            display_scatterplot_3D(model, user_input, similar_word, labels, color_map, annotation, dim_red, perplexity, learning_rate, iteration, top_n, sample_n)

        st.subheader('각 입력에 대한 상위 5개의 가장 유사한 단어')
        count = 0
        for i in range(len(user_input)):
            st.write(str(user_input[i])+'과 가장 유사한 단어는 아래와 같습니다.')
            horizontal_bar(similar_word[count:count+5], similarity[count:count+5])

            count += top_n

with tab2:
    st.subheader('문서 내 토픽 모델링(LDA) 시각화를 위한 웹앱입니다.')
    col1_tab2, col2_tab2 = st.columns([1, 4])
    with col1_tab2:
        num_topics = st.slider(
            '⁜ 토픽의 수를 설정하세요.',
            3, 10, (5)
        )
        passes = st.slider(
            '⁜ LDA모델 훈련 횟수를 설정하세요.',
            1, 100, (50)
        )
    lda = LDA_Model()
    lda_model, corpus, dictionary, doc_list = lda.get_lda_model(df, company_name, year, col_dic[col], num_topics, passes)

    with col2_tab2:
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        st.write(f"{year} {company_name} {col} 토픽 모델링 - LDA")
        html_string = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(html_string, width=1300, height=800, scrolling=True)

with tab3:
    st.subheader('어휘의 빈도에 따른 히트맵 시각화를 위한 웹앱입니다.')
    col1_tab3, col2_tab3 = st.columns([1, 4])
    with col1_tab3:
        word_n = st.slider(
            '히트맵에 보여줄 단어의 수를 선택하세요.',
            5, 30, (15)
        )
        height = word_n * 53

    @st.experimental_memo
    def display_heatmap(df, company_name, col, word_n):
        df_join = funcs.get_all_most_common_join_df(df, company_name, col, word_n)
        plot_figure = px.imshow(
            df_join.iloc[:, 1:],
            text_auto=True,
            aspect="auto",
            height=height,
        )
        plot_figure.update_xaxes(side="top")
        st.plotly_chart(plot_figure)

    with col2_tab3:
        st.markdown(f"**{year} {company_name} 연도별 TOP{word_n} 빈출 단어 히트맵**")
        display_heatmap(df, company_name, col_dic[col], word_n)

with tab4:
    st.subheader('연도별 평점 트렌드 분석 시각화를 위한 웹앱입니다.')
    st.markdown(f'**[{company_name}]**')
    df_comp = funcs.get_comp(df, company_name)
    fields = ['Ratings', 'Culture', 'WorkLifeBalance', 'Benefits', 'Management', 'Opportunity']
    fields2 = ['Potential', 'Recommend']
    field_dic = {
        'Ratings': '전체 평점',
        'Culture': '사내 문화',
        'WorkLifeBalance': '워라벨',
        'Benefits': '복지/복리후생',
        'Management': '경영진',
        'Opportunity': '승진/발전 기회',
        'Potential': '회사 발전가능성',
        'Recommend': '추천 여부'
    }

    cols1_tab4 = st.columns(2)
    cols2_tab4 = st.columns(2)
    cols3_tab4 = st.columns(2)
    for i, field in enumerate(fields):
        with st.container():
            years, trends = funcs.get_fluctuation(df_comp, field)
            fig1 = plt.figure(figsize=(7, 2))
            sns.barplot(x=years, y=trends, palette='crest')
            if i < 2:
                with cols1_tab4[i]:
                    st.markdown(f'[{field_dic[field]}] 연도별 트렌드')
                    st.pyplot(fig1)
            elif i > 1 and i < 4:
                with cols2_tab4[i-2]:
                    st.markdown(f'[{field_dic[field]}] 연도별 트렌드')
                    st.pyplot(fig1)
            elif i > 3:
                with cols3_tab4[i-4]:
                    st.markdown(f'[{field_dic[field]}] 연도별 트렌드')
                    st.pyplot(fig1)
    
    cols4_tab4 = st.columns(2)
    for i, field in enumerate(fields2):
        with st.container():
            years, trends = funcs.get_fluctuation2(df_comp, field)
            fig2 = plt.figure(figsize=(7, 2))
            sns.barplot(x=years, y=trends, palette='flare')
            with cols4_tab4[i]:
                st.markdown(f'[{field_dic[field]}] 연도별 트렌드')
                st.pyplot(fig2)
