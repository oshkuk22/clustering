import os
import pandas as pd
from gensim.utils import tokenize
from gensim import corpora, models, similarities
import pymorphy2
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans


def get_stopwords():
    if os.path.isfile(os.path.join(os.path.abspath(os.curdir), 'stopwords.txt')):
        try:
            with open(os.path.join(os.path.abspath(os.curdir), 'stopwords.txt'), 'r') as text_file:
                text_read_file = text_file.read().split('\n')

            return text_read_file
        except Exception as e:
            print(str(e))
    else:
        message = u'Файл стоп слов не найден'
        print(message)


def get_cloud(tokens_):
    try:
        if not tokens_:
            print('Невозможно построить облако из 0 слов')
        else:
            wordcloud = WordCloud(width=1024, height=1024, random_state=1, background_color='black',
                                  colormap='Set2', collocations=False, stopwords=STOPWORDS).generate(tokens_)
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            plt.imshow(wordcloud)
            plt.show()
    except Exception as e:
        print(str(e))


def get_lemmas(tokens_no_stop_words):
    morph = pymorphy2.MorphAnalyzer(path='pymorphy2_dicts_ru/data')
    lemma_tokens = [morph.parse(token)[0].normal_form for token in tokens_no_stop_words]
    return lemma_tokens


def tokenize_in_df(string_):
    try:
        return list(tokenize(string_, lowercase=True, deacc=True, ))
    except Exception as e:
        print(str(e))
        return ""


def del_stopwords(lst_tokens_):
    stopwords = get_stopwords()
    list_no_stopwords = [token_ for token_ in lst_tokens_ if token_ not in stopwords]
    return list_no_stopwords


data_frame_ = pd.read_excel('famous_autor.xlsx')
texts_to_compare = list(data_frame_.head(5)['texts'])
authors = data_frame_['authors']
texts = data_frame_['texts']
data_frame_['tokens'] = texts.apply(tokenize_in_df)
data_frame_['tokens_lem'] = data_frame_['tokens'].apply(get_lemmas)
data_frame_['tokens_lem_no_sw'] = data_frame_['tokens_lem'].apply(del_stopwords)

dictionary = corpora.Dictionary(data_frame_["tokens_lem_no_sw"])
feature_cnt = len(dictionary.token2id)
print(dictionary.token2id)


lst__ = list(data_frame_['tokens_lem_no_sw'])
lst_res = []

for list_ in lst__:
    lst_res.extend(list_)

for_cloud = " ".join(lst_res)

get_cloud(for_cloud)

corpus = [dictionary.doc2bow(text) for text in data_frame_['tokens_lem_no_sw']]

tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)

for text in texts_to_compare:
    kw_vector = dictionary.doc2bow(tokenize(text))
    data_frame_[text] = index[tfidf[kw_vector]]


data_frame_.to_excel('tfidf.xlsx')

print(data_frame_[text])

# clustering = DBSCAN(eps=0.51, metric='cosine', min_samples=3).fit(data_frame_[text])
# print(clustering.labels_)