import pandas as pd
import re
import os
import string
from razdel import tokenize
from PyQt5 import QtCore, uic, QtWidgets
from nltk import ngrams
from collections import Counter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
import pickle
import numpy as np
import scipy.spatial.distance as ds
from nltk import sent_tokenize, word_tokenize
import natasha
import pymystem3
import pymorphy2
import matplotlib.pyplot as plt


N_GRAMS_CHARACTERS_DVMA = [('*', '*', '*'), (' ', '*', '*'), ('*', '*', ' '), ('#', '#', '#'), ('*', ' ', '*'),
                           ('*', '*', ','), ('*', ',', ' '), ('*', '*', '.'), (' ', 'н', 'а'), (',', ' ', '*'),
                           (' ', 'и', ' '), (' ', '#', '#'), ('*', '.', ' '), (' ', 'в', ' '), (' ', 'в', 'а'),
                           ('и', ' ', '*'), (' ', 'я', ' '), ('т', 'е', ' '), ('*', ' ', 'в'), ('я', ' ', '*'),
                           (' ', 'м', 'о'), ('*', ' ', 'и'), (' ', 'в', 'ы'), ('н', 'а', ' '), ('н', 'е', ' '),
                           ('#', '#', ' '), ('в', 'ы', ' '), (' ', 'н', 'е'), ('.', ' ', '*'), ('и', 'т', 'е'),
                           ('т', 'о', ' '), ('м', 'ы', ' '), ('е', ' ', '*'), (' ', 'п', 'о'), (',', ' ', '#'),
                           (' ', 'м', 'ы'), ('а', ' ', '*'), (' ', 'з', 'а'), ('#', ',', ' '), ('ы', ' ', '*'),
                           ('т', 'ь', ' '), ('о', 'й', ' '), ('л', 'и', ' '), ('а', 'н', 'и'), ('е', 'р', 'е'),
                           ('*', ' ', 'с'), ('#', '#', ','), ('*', ' ', 'н'), ('е', 'м', ' '), ('в', 'а', 'ш'),
                           ('б', 'у', 'д'), ('т', ' ', '*'), (' ', 'б', 'у'), ('!', '!', '!'), ('а', 'м', ' '),
                           ('й', ' ', '*'), ('о', 'р', 'о'), ('п', 'е', 'р'), (' ', 'ч', 'т'), ('ч', 'т', 'о'),
                           (' ', 'п', 'р'), (' ', 'м', 'н'), (' ', 'п', 'е'), (' ', 'э', 'т'), ('а', 'с', ' '),
                           ('е', ' ', 'с'), ('е', 'т', 'е'), ('н', 'я', ' '), ('е', 'т', ' '), (' ', 'к', 'о'),
                           ('м', 'е', 'н'), ('в', 'а', 'с'), (' ', 'т', 'о'), ('о', ' ', '*'), (' ', 'с', 'о'),
                           ('е', ' ', 'в'), ('о', 'т', ' '), ('с', 'т', 'р'), ('м', ' ', '*'), (' ', 'с', 'т'),
                           ('э', 'т', 'о'), ('и', 'х', ' '), ('е', 'й', ' '), ('ж', 'е', ' '), (' ', 'с', 'в'),
                           ('в', ' ', '*'), ('х', ' ', '*'), ('е', 'г', 'о'), (' ', 'р', 'у'), (' ', 'о', 'т'),
                           (' ', 'в', 'о'), ('е', ' ', 'н'), (' ', 'г', 'о'), (' ', 'т', 'а'), (' ', 'в', 'с'),
                           ('в', 'а', 'м'), (' ', 'б', 'о'), ('т', 'а', 'к'), ('д', 'и', 'т'), ('*', '*', '!'),
                           ('г', 'о', ' '), ('а', 'т', 'ь'), ('*', ' ', 'п'), ('*', ' ', 'у'), ('в', 'с', 'е'),
                           ('у', 'д', 'е'), (' ', 'м', 'е'), ('и', ' ', 'в'), ('т', 'о', 'р'), ('р', 'е', 'в'),
                           ('р', 'о', 'д'), (' ', 'т', 'е'), (' ', 'к', 'а'), ('к', 'о', 'л'), ('*', ' ', 'м'),
                           ('д', 'е', 'т'), ('о', 'т', 'о'), ('а', 'к', ' '), ('у', ' ', '*'), ('#', ' ', '*'),
                           ('у', ' ', 'в'), ('д', 'а', 'н'), ('с', 'т', 'а'), ('е', 'н', 'и'), ('м', 'о', 'й'),
                           ('в', 'о', 'д'), (',', ' ', 'в'), ('т', 'р', 'а'), ('м', 'н', 'е'), (' ', 'с', 'у'),
                           ('о', ' ', 'в'), ('о', 'е', ' '), ('е', ' ', 'м'), ('.', ' ', 'я'), ('о', 'д', 'и'),
                           ('т', 'и', ' '), ('п', 'р', 'и'), ('м', 'е', 'р'), ('л', 'е', 'й'), ('е', 'с', 'л'),
                           ('и', 'т', 'ь'), ('с', 'в', 'о'), (' ', 'с', 'е'), ('с', 'я', ' '), ('с', 'т', 'в'),
                           ('и', 'е', ' '), ('р', 'у', 'б'), ('у', 'б', 'л'), ('б', 'л', 'е'), ('г', 'о', 'р'),
                           ('.', ' ', 'м'), ('с', 'л', 'и'), ('б', 'ы', ' '), ('а', 'ш', 'е'), (' ', 'т', 'р'),
                           ('т', 'р', 'е'), ('у', 'ю', ' '), (' ', 'е', 'с'), ('*', '.', ','), ('.', ',', ' '),
                           ('*', ' ', 'о'), ('ь', ' ', '*'), ('е', 'х', ' '), ('я', ' ', 'в'), ('и', ' ', 'с'),
                           ('у', 'т', ' '), (' ', '#', ' '), (',', ' ', 'м'), ('е', 'с', 'т'), ('е', 'н', 'я'),
                           ('*', '*', ':'), ('к', 'о', 'т'), ('и', ' ', 'п'), ('б', 'о', 'м'), (' ', 'с', 'л'),
                           ('а', 'я', ' '), (' ', 'а', ' '), ('а', 'с', 'т')]


N_GRAMS_CHARACTERS_DVSA = [(' ', '*', ' '), ('*', ' ', '*'), ('*', ',', ' '), (' ', '*', ','), (' ', '*', '.'),
                           (' ', 'н', 'а'), (',', ' ', '*'), (' ', 'и', ' '), ('*', '.', ' '), (' ', 'в', ' '),
                           (' ', 'в', 'а'), ('и', ' ', '*'), (' ', 'я', ' '), ('т', 'е', ' '), ('*', ' ', 'в'),
                           ('я', ' ', '*'), (' ', 'м', 'о'), ('*', ' ', 'и'), (' ', 'в', 'ы'), ('н', 'а', ' '),
                           (' ', '#', ' '), ('н', 'е', ' '), ('в', 'ы', ' '), (' ', 'н', 'е'), ('.', ' ', '*'),
                           ('и', 'т', 'е'), ('т', 'о', ' '), ('м', 'ы', ' '), ('е', ' ', '*'), (' ', 'п', 'о'),
                           (',', ' ', '#'), (' ', 'м', 'ы'), ('а', ' ', '*'), (' ', 'з', 'а'), ('#', ',', ' '),
                           ('ы', ' ', '*'), ('т', 'ь', ' '), (' ', '#', ','), ('о', 'й', ' '), ('л', 'и', ' '),
                           ('а', 'н', 'и'), ('е', 'р', 'е'), ('*', ' ', 'с'), ('*', ' ', 'н'), ('е', 'м', ' '),
                           ('в', 'а', 'ш'), ('б', 'у', 'д'), ('т', ' ', '*'), (' ', 'б', 'у'), ('!', '!', '!'),
                           ('а', 'м', ' '), ('й', ' ', '*'), ('о', 'р', 'о'), ('п', 'е', 'р'), (' ', 'ч', 'т'),
                           ('ч', 'т', 'о'), (' ', 'п', 'р'), (' ', 'м', 'н'), (' ', 'п', 'е'), (' ', 'э', 'т'),
                           ('а', 'с', ' '), ('е', ' ', 'с'), ('е', 'т', 'е'), ('н', 'я', ' '), ('е', 'т', ' '),
                           (' ', 'к', 'о'), ('м', 'е', 'н'), ('в', 'а', 'с'), (' ', 'т', 'о'), ('о', ' ', '*'),
                           (' ', 'с', 'о'), ('е', ' ', 'в'), ('о', 'т', ' '), ('с', 'т', 'р'), ('м', ' ', '*'),
                           (' ', 'с', 'т'), ('э', 'т', 'о'), ('и', 'х', ' '), ('е', 'й', ' '), ('ж', 'е', ' '),
                           (' ', 'с', 'в'), ('в', ' ', '*'), ('х', ' ', '*'), ('е', 'г', 'о'), (' ', 'р', 'у'),
                           (' ', 'о', 'т'), (' ', 'в', 'о'), ('е', ' ', 'н'), (' ', 'г', 'о'), (' ', 'т', 'а'),
                           (' ', 'в', 'с'), ('в', 'а', 'м'), (' ', 'б', 'о'), ('т', 'а', 'к'), ('д', 'и', 'т'),
                           (' ', '*', '!'), ('г', 'о', ' '), ('а', 'т', 'ь'), ('*', ' ', 'п'), ('*', ' ', 'у'),
                           ('в', 'с', 'е'), ('у', 'д', 'е'), (' ', 'м', 'е'), ('и', ' ', 'в'), ('т', 'о', 'р'),
                           ('р', 'е', 'в'), ('р', 'о', 'д'), (' ', 'т', 'е'), (' ', 'к', 'а'), ('к', 'о', 'л'),
                           ('*', ' ', 'м'), ('д', 'е', 'т'), ('о', 'т', 'о'), ('а', 'к', ' '), ('у', ' ', '*'),
                           ('#', ' ', '*'), ('у', ' ', 'в'), ('д', 'а', 'н'), ('с', 'т', 'а'), ('е', 'н', 'и'),
                           ('м', 'о', 'й'), ('в', 'о', 'д'), (',', ' ', 'в'), ('т', 'р', 'а'), ('м', 'н', 'е'),
                           (' ', 'с', 'у'), ('о', ' ', 'в'), ('о', 'е', ' '), ('е', ' ', 'м'), ('.', ' ', 'я'),
                           ('о', 'д', 'и'), ('т', 'и', ' '), ('п', 'р', 'и'), ('м', 'е', 'р'), ('л', 'е', 'й'),
                           ('е', 'с', 'л'), ('и', 'т', 'ь'), ('с', 'в', 'о'), (' ', 'с', 'е'), ('с', 'я', ' '),
                           ('с', 'т', 'в'), ('и', 'е', ' '), ('р', 'у', 'б'), ('у', 'б', 'л'), ('б', 'л', 'е'),
                           ('г', 'о', 'р'), ('.', ' ', 'м'), ('с', 'л', 'и'), ('б', 'ы', ' '), ('а', 'ш', 'е'),
                           (' ', 'т', 'р'), ('т', 'р', 'е'), ('у', 'ю', ' '), (' ', 'е', 'с'), ('*', '.', ','),
                           ('.', ',', ' '), ('*', ' ', 'о'), ('ь', ' ', '*'), ('е', 'х', ' '), ('я', ' ', 'в'),
                           ('и', ' ', 'с'), ('у', 'т', ' '), (',', ' ', 'м'), ('е', 'с', 'т'), ('е', 'н', 'я'),
                           ('к', 'о', 'т'), ('и', ' ', 'п'), ('б', 'о', 'м'), (' ', 'с', 'л'), ('а', 'я', ' '),
                           (' ', 'а', ' '), ('а', 'с', 'т')]


def pca_func(matrix, n_comp=3):

    pca = TruncatedSVD(n_components=n_comp)

    pca_comp = pca.fit_transform(matrix)

    std_ = np.sum(pca.explained_variance_ratio_)

    with open(os.path.join('pca', 'pca_' + '.pickle'), 'wb') as file:
        pickle.dump(pca, file)


def pca_use(matrix):
    with open(os.path.join('pca', 'pca_' + '.pickle'), 'rb') as file:
        pca_ = pickle.load(file)
    pca_comp = pca_.transform(matrix)
    std_pca = np.sum(pca_.explained_variance_ratio_)
    print(std_pca)
    return pca_comp #, std_pca


def message_info(message):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle(u"Информация")
    msg.show()
    msg.exec_()


def freq_dvma_dvsa_characters(text_distruct, type_destruction='dvma'):
    list_result = []
    n_grams_characters = Counter(ngrams(text_distruct, 3)).most_common()
    if type_destruction == 'dvma':
        n_grams_ = N_GRAMS_CHARACTERS_DVMA
    elif type_destruction == 'dvsa':
        n_grams_ = N_GRAMS_CHARACTERS_DVSA
    for elem in n_grams_:
        flag_yes = False
        for elem_counter in n_grams_characters:
            if elem == elem_counter[0]:
                list_result.append('_'.join(elem) + '-' + str(round(elem_counter[1] / len(text_distruct), 4)))
                flag_yes = True
        if not flag_yes:
            list_result.append('_'.join(elem) + '-' + str(0))
    return list_result


def get_most_popular_rus_words_top_sharov():
    if os.path.isfile(os.path.join(os.path.abspath(os.curdir), 'infotext', 'top_4926._sharov.txt')):
        try:
            with open(os.path.join(os.path.abspath(os.curdir), 'infotext', 'top_4926._sharov.txt'), 'r') as text_file:
                most_popular_rus_words_top_10000 = text_file.read().split('\n')
            return most_popular_rus_words_top_10000
        except Exception as e:
            message_info(str(e))
    else:
        message = u'Файл стоп слов не найден'
        message_info(message)


def get_tokens(text):
    tokens_list = []
    positional = []
    tokens_ = list(tokenize(text))
    for i, j in enumerate(tokens_):
        tokens_list.append(tokens_[i].text)
        positional.append([tokens_[i].start, tokens_[i].stop])
    return positional, tokens_list


def is_digit(token):
    if token.isdigit():
        return True
    else:
        try:
            float(token)
            return True
        except ValueError:
            return False


def text_distortion_dvma(text, most_popular):
    morph = pymorphy2.MorphAnalyzer(path='pymorphy2_dicts_ru/data')
    distortion_text = ''
    pos, tokens = get_tokens(text)
    for ind, token in enumerate(tokens):
        if morph.parse(token.lower())[0].normal_form not in most_popular:
            if is_digit(token):
                for dig in token:
                    if dig not in string.punctuation:
                        distortion_text += '#'
                    else:
                        distortion_text += dig
                if pos[ind][1] != len(text):
                    if text[pos[ind][1]] == ' ':
                        distortion_text += ' '
            elif token in string.punctuation or token[0] in string.punctuation + "«" + "»" + "—":
                distortion_text += token
                if pos[ind][1] != len(text):
                    if text[pos[ind][1]] == ' ':
                        distortion_text += ' '

            else:
                distortion_text += '*' * len(token)
                if pos[ind][1] != len(text):
                    if text[pos[ind][1]] == ' ':
                        distortion_text += ' '
        else:
            distortion_text += token
            if pos[ind][1] != len(text):
                if text[pos[ind][1]] == ' ':
                    distortion_text += ' '

    return distortion_text


def text_distortion_dvsa(text, most_popular):
    morph = pymorphy2.MorphAnalyzer(path='pymorphy2_dicts_ru/data')
    distortion_text = ''
    pos, tokens = get_tokens(text)
    for ind, token in enumerate(tokens):
        if morph.parse(token.lower())[0].normal_form not in most_popular:
            if is_digit(token):
                distortion_text += '#'
                if pos[ind][1] != len(text):
                    if text[pos[ind][1]] == ' ':
                        distortion_text += ' '
            elif token in string.punctuation or token[0] in string.punctuation + "«" + "»" + "—":
                distortion_text += token
                if pos[ind][1] != len(text):
                    if text[pos[ind][1]] == ' ':
                        distortion_text += ' '

            else:
                distortion_text += '*'
                if pos[ind][1] != len(text):
                    if text[pos[ind][1]] == ' ':
                        distortion_text += ' '
        else:
            distortion_text += token
            if pos[ind][1] != len(text):
                if text[pos[ind][1]] == ' ':
                    distortion_text += ' '
    return distortion_text


top_words_sharov = get_most_popular_rus_words_top_sharov()

data_frame_ = pd.read_excel('famous_autor.xlsx')
authors = data_frame_['authors']
texts = data_frame_['texts']
lst_texts = []
lst_dvma = []
lst_dvsa = []
list_n_grams = []
len_massage = []
matrix_features = []

for i in texts:
    vector_features = []
    len_massage.append(len(i))
    text_digital = re.sub(r'[a-zA-Z]', r'', i.replace('\n', '').replace('\t', '').replace('_', '').lower().strip())
    # lst_texts.append(re.sub(r'[\d+]', r'', text_digital.strip()))
    lst_texts.append(text_digital)
    el_dvma = text_distortion_dvma(text_digital, top_words_sharov)
    lst_dvma.append(el_dvma)
    el_dvsa = text_distortion_dvsa(text_digital, top_words_sharov)
    lst_dvsa.append(el_dvsa)

    list_dvma_characters = freq_dvma_dvsa_characters(el_dvma, type_destruction='dvma')
    for elem in list_dvma_characters:
        vector_features.append(float(elem.split('-')[-1]))

    list_dvsa_characters = freq_dvma_dvsa_characters(el_dvsa, type_destruction='dvsa')
    for elem in list_dvsa_characters:
        vector_features.append(float(elem.split('-')[-1]))

    matrix_features.append(np.array(vector_features))

pca_func(matrix_features, n_comp=3)
pca_comp = pca_use(matrix_features)

clustering = DBSCAN(eps=0.51, metric='cosine', min_samples=3).fit(pca_comp)
print(clustering.labels_)

clustering = KMeans(n_clusters=15).fit(pca_comp)
print(clustering.labels_)
print(clustering.cluster_centers_)

fig = plt.figure(figsize=(13, 7))
plt.gcf().canvas.manager.set_window_title('Растояние между векторами PCA в пространстве')

ax_3d = fig.add_subplot(projection='3d')

ax_3d.scatter(pca_comp[:, 0], pca_comp[:, 1], pca_comp[:, 2], color='blue', label='text')
ax_3d.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1],
              clustering.cluster_centers_[:, 2], color='red', label='center_clusters')
ax_3d.set_xlabel('PC1')
ax_3d.set_ylabel('PC2')
ax_3d.set_zlabel('PC3')
ax_3d.legend()
plt.show()


#     pos, tokens = get_tokens(el_dvma)
#
#     n_grams_text_tokens = ngrams(el_dmsa, 3)
#
#     for elem in n_grams_text_tokens:
#         list_n_grams.append(elem)
#
# list_n_grams_counter = Counter(list_n_grams).most_common()
#
# list_n_grams_total = [i[0] for i in list_n_grams_counter if i[1] > 30]
#
# print(len_massage)
# print(list_n_grams_total)
#
# with open(os.path.join('list_cont_n_grams_dmsa_characters' + '.txt'), 'w') as file:
#     file.write(str(list_n_grams_total))

# print(lst_dvsa)
