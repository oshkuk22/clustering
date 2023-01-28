import os
import re
import pandas as pd
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.cluster import DBSCAN, KMeans


def pca_use(matrix):
    with open(os.path.join('pca', 'pca_' + '.pickle'), 'rb') as file:
        pca_ = pickle.load(file)
    pca_comp = pca_.transform(matrix)
    std_pca = np.sum(pca_.explained_variance_ratio_)
    print(std_pca)
    return pca_comp #, std_pca


def pca_func(matrix, n_comp=3):
    pca = TruncatedSVD(n_components=n_comp)
    pca_comp = pca.fit_transform(matrix)
    std_ = np.sum(pca.explained_variance_ratio_)
    with open(os.path.join('pca', 'pca_' + '.pickle'), 'wb') as file:
        pickle.dump(pca, file)


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
        return list(tokenize(string_, lowercase=True, deacc=True))
    except Exception as e:
        print(str(e))
        return ""


def del_stopwords(lst_tokens_):
    stopwords = get_stopwords()
    list_no_stopwords = [token_ for token_ in lst_tokens_ if token_ not in stopwords]
    return list_no_stopwords


def tokinizer_(data_frame):
    texts = data_frame['texts']
    data_frame['tokens'] = texts.apply(tokenize_in_df)
    data_frame['tokens_lem'] = data_frame['tokens'].apply(get_lemmas)
    data_frame['tokens_lem_no_sw'] = data_frame['tokens_lem'].apply(del_stopwords)
    return data_frame


data_frame = pd.read_excel('anonim.xlsx')
# texts_to_compare = list(data_frame.head(5)['texts'])
authors = data_frame['authors']

data_frame = tokinizer_(data_frame)
lst__ = list(data_frame['tokens_lem_no_sw'])

lst_res = []
list_tfidf = []

for list_ in lst__:
    list_tfidf.append(str(" ".join(list_)))
    lst_res.extend(list_)


print(list_tfidf)
print('________________')
for_cloud = " ".join(lst_res)
print('________________')
get_cloud(for_cloud)
model_tfidf = TfidfVectorizer(ngram_range=(2, 5), analyzer='char', tokenizer=str.split)#, max_features=100)
bag_of_words = model_tfidf.fit_transform(list_tfidf)
print(bag_of_words.toarray())
features_names = model_tfidf.get_feature_names_out()
print(pd.DataFrame(bag_of_words.toarray(), columns=features_names))

pca_func(bag_of_words.toarray(), n_comp=4)
pca_comp = pca_use(bag_of_words.toarray())

print(pca_comp.shape)

clustering = DBSCAN(eps=2, metric='cosine', min_samples=1).fit(pca_comp)
print(clustering.labels_)

clustering = KMeans(n_clusters=9).fit(pca_comp)
data_frame_for_clor_map = pd.DataFrame()
centroid = clustering.cluster_centers_
cen_x = [i[0] for i in centroid]
cen_y = [i[1] for i in centroid]
cen_z = [i[2] for i in centroid]
data_frame_for_clor_map['clusters'] = clustering.labels_
data_frame_for_clor_map['cen_x'] = data_frame_for_clor_map['clusters'].map({0: cen_x[0], 1: cen_x[1], 2: cen_x[2],
                                                                            3: cen_x[3], 4: cen_x[4], 5: cen_x[5],
                                                                            6: cen_x[6], 7: cen_x[7], 8: cen_x[8]})
data_frame_for_clor_map['cen_y'] = data_frame_for_clor_map['clusters'].map({0: cen_y[0], 1: cen_y[1], 2: cen_y[2],
                                                                            3: cen_y[3], 4: cen_y[4], 5: cen_y[5],
                                                                            6: cen_y[6], 7: cen_y[7], 8: cen_y[8]})
data_frame_for_clor_map['cen_z'] = data_frame_for_clor_map['clusters'].map({0: cen_z[0], 1: cen_z[1], 2: cen_z[2],
                                                                            3: cen_z[3], 4: cen_z[4], 5: cen_z[5],
                                                                            6: cen_z[6], 7: cen_z[7], 8: cen_z[8]})

colors = ['yellow', 'blue', 'green', 'black', 'cyan', 'magenta', 'olive', 'purple', 'orange']

data_frame_for_clor_map['c'] = data_frame_for_clor_map['clusters'].map({0: colors[0], 1: colors[1], 2: colors[2],
                                                                        3: colors[3], 4: colors[4], 5: colors[5],
                                                                        6: colors[6], 7: colors[7], 7: colors[7],
                                                                        8: colors[8]})

fig = plt.figure(figsize=(13, 7))
plt.gcf().canvas.manager.set_window_title('Растояние между векторами PCA в пространстве')

ax_3d = fig.add_subplot(projection='3d')

ax_3d.scatter(pca_comp[:, 0], pca_comp[:, 1], pca_comp[:, 2], color=data_frame_for_clor_map.c, label='text')
ax_3d.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1],
              clustering.cluster_centers_[:, 2], color='red', label='center_clusters')
ax_3d.set_xlabel('PC1')
ax_3d.set_ylabel('PC2')
ax_3d.set_zlabel('PC3')
ax_3d.legend()

for label, x, y, z in zip(authors, pca_comp[:, 0], pca_comp[:, 1], pca_comp[:, 2]):
    ax_3d.text(x, y, z, '%s' % (label), size=6, zorder=1, color='k', ha='right',
               va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

plt.show()

plt.scatter(pca_comp[:, 1], pca_comp[:, 2], color=data_frame_for_clor_map.c, label='text')
plt.scatter(clustering.cluster_centers_[:, 1], clustering.cluster_centers_[:, 2], color='red', label='center_clusters')
for label, x, y in zip(authors, pca_comp[:, 1], pca_comp[:, 2]):
    plt.annotate(label, xy=(x, y), xytext=(-5, 5), textcoords='offset points', size=6, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.01', fc='yellow', alpha=0.3))
plt.show()
print(clustering.labels_)
print(clustering.inertia_)
# clustering = DBSCAN(eps=0.51, metric='cosine', min_samples=3).fit(data_frame_[text])
# print(clustering.labels_)