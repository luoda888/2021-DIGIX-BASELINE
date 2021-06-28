from gensim.summarization import bm25
from gensim import corpora
import pandas as pd
from nltk.stem import WordNetLemmatizer
# from snowballstemmer import TurkishStemmer
from TurkishStemmer import TurkishStemmer
from itertools import islice
import nltk
import csv
import os
import glob
import heapq
import time
import re

DATA_DIR = "data"
OUTPUT_FILE_EN = 'en_recall_10.csv'
OUTPUT_FILE_TR = 'tr_recall_10.csv'
NUM_PAGES_RESULT = 10
NUM_PAGES_PREDICT = 10

wnl = WordNetLemmatizer()
turkStem = TurkishStemmer()

# STOP_WORDS = ['<br>', 'I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours',
#               'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
#               'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
#               'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
#               'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
#               'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'util',
#               'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
#               'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
#               'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
#               'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
#               'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
#               'too', 'very', 'can', "can't", 'will', 'just', "don't", 'should', "shouldn't", 'now',
#               "aren't", "didn't", "haven't", "mustn't", "wasn't", "weren't", "wouldn't", "needn't",
#               "won't"]

def getWordsFromURL(url, lang):
    words_list = re.compile(r'[\:/?=\-&.,_@%!$0123456789()&*+\[\]]+',re.UNICODE).split(url)
    drop_words = set(['', 'http', 'https', 'www', 'com', '\t', 'm', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'])

    return [turkStem.stem(word.lower()) for word in words_list if word.lower() not in drop_words]
    # return [simplemma.lemmatize(word.lower(), langdata, greedy=True) for word in words_list if word.lower() not in drop_words]

def sentence_processing(sentence):
    tokens = sentence.lower().replace('<br>', ' ')\
                        .replace('"', ' ') \
                        .replace('\'', ' ') \
                        .replace('.', ' ') \
                        .replace(',', ' ') \
                        .replace('?', ' ') \
                        .replace('!', ' ') \
                        .replace('[', ' ') \
                        .replace(']', ' ') \
                        .replace('(', ' ') \
                        .replace(')', ' ') \
                        .replace('{', ' ') \
                        .replace('}', ' ') \
                        .replace('<', ' ') \
                        .replace('>', ' ') \
                        .replace(':', ' ') \
                        .replace('\\', ' ') \
                        .replace('`', ' ') \
                        .replace('=', ' ') \
                        .replace('$', ' ') \
                        .replace('/', ' ') \
                        .replace('*', ' ') \
                        .replace(';', ' ') \
                        .replace('\/', ' ') \
                        .replace('-', ' ') \
                        .replace('^', ' ') \
                        .replace('|', ' ') \
                        .replace('%', ' ').split()

    return tokens

def tr_process(input_str):
    new_str = input_str.lower().replace('ı', 'i') \
                               .replace('ğ', 'g') \
                               .replace('ç', 'c') \
                               .replace('ö', 'o') \
                               .replace('ş', 's') \
                               .replace('ü', 'u')
    return new_str

def build_bm25_model(data_dir, language):
    corpus = []
    urls = []
    files = glob.glob(os.path.join(data_dir, '*'))
    idx = 0
    for file in files:
        with open(file, encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                parts = line.split('\x01')[:3]
                if len(parts) == 1:
                    title_with_content = ''
                elif len(parts) == 2:
                    title_with_content = parts[1]
                else:
                    title_with_content = (parts[1] + ' ') * 20 + '.' + parts[2]
                if language == 'en':
                    corpus.append([wnl.lemmatize(word) for word in sentence_processing(title_with_content)]
                                + 20 * getWordsFromURL(parts[0], language))
                if language == 'tr':
                    # corpus.append(sentence_processing(title_with_content) + 20 * getWordsFromURL(parts[0]))
                    corpus.append([turkStem.stem(word) for word in sentence_processing(title_with_content)]
                                  + 20 * getWordsFromURL(parts[0], language))
                    # corpus.append([simplemma.lemmatize(word, langdata, greedy=True) for word in
                    #                sentence_processing(title_with_content)] + 20 * getWordsFromURL(parts[0]))
                urls.append(parts[0])

        idx += 1
        if idx % 1000 == 0:
            print (idx)
    bm25_model = bm25.BM25(corpus)
    print ("The corpus size is : ", len(corpus))
    return corpus, urls, bm25_model

def get_data(file, result_num):
    qid_dict = {}
    result_dict = {}
    df = pd.DataFrame()
    qid = []
    query = []
    doc = []
    rank = []
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in islice(reader, 1, None):
            qid.append(line[0])
            query.append(line[1])
            doc.append(line[2])
            rank.append(line[3])
            qid_dict[line[0]] = line[1]
            if line[0] not in result_dict:
                result_dict[line[0]] = [line[2]]
            elif len(result_dict[line[0]]) < result_num:
                result_dict[line[0]].append(line[2])
            else:
                continue
    df['qid'] = qid
    df['query'] = query
    df['doc'] = doc
    df['rank'] = rank
    print (len(set(df['qid'].tolist())))
    return df, qid_dict, result_dict

def get_top_pages(model, num_pages, df, urls, result_dict, language, qid_dict):
    qids = set(df['qid'].tolist())
    predict_dict = {}
    score_dict = {}
    idx = 0
    for qid in qids:
        query = qid_dict[qid]
        if language == 'tr':
            new_query = query + ' ' + tr_process(query)
            # new_query = query
            query_tokens = [turkStem.stem(token) for token in sentence_processing(new_query)]
            # query_tokens = [simplemma.lemmatize(token, langdata, greedy=True) for token in sentence_processing(new_query)]
        if language == 'en':
            query_tokens = [wnl.lemmatize(token) for token in sentence_processing(query)]
        scores = model.get_scores(query_tokens)
        indices = sorted(range(len(scores)), key=lambda k: scores[k])[::-1][:num_pages]
        pages = [urls[i] for i in indices]
        predict_dict[qid] = pages
        score_dict[qid] = [scores[i] for i in indices]
        idx += 1
        if idx % 100 == 0:
            print (idx)
    score = 0
    idx = 0
    for qid in predict_dict:
        true_recall = 0
        for page in predict_dict[qid]:
            if page in result_dict[qid]:
                true_recall += 1
        score += true_recall / NUM_PAGES_RESULT
        idx += 1
        if idx % 100 == 0:
            print (idx)
    return score / len(predict_dict), predict_dict, score_dict

def write_recall_to_csv(qid_dict, predict_dict, score_dict, output_file):
    qid_result = []
    query_result = []
    page_result = []
    score_result = []
    for qid in predict_dict:
        query = qid_dict[qid]
        pages = predict_dict[qid]
        scores = score_dict[qid]
        for idx, page in enumerate(pages):
            qid_result.append(qid)
            query_result.append(query)
            page_result.append(page)
            score_result.append(scores[idx])
    result_df = pd.DataFrame()
    result_df['qid'] = qid_result
    result_df['query'] = query_result
    result_df['page'] = page_result
    result_df['score'] = score_result
    result_df.to_csv(output_file, encoding='utf-8', index=False)

if __name__ == "__main__":
    en_corpus, en_urls, en_bm25_model = build_bm25_model(os.path.join(DATA_DIR, "en_corpus"), 'en')
    print ("en corpus processing finished")
    df_en_train, qid_dict_en_train, result_dict_en_train = \
        get_data(os.path.join(DATA_DIR, "train", "train_en.tsv"), NUM_PAGES_RESULT)
    print ("en train processing finished")
    df_en_test, qid_dict_en_test, result_dict_en_test = \
        get_data(os.path.join(DATA_DIR, "test", "test_en.tsv"), NUM_PAGES_RESULT)
    print ("en test processing finished")
    en_train_result, en_train_predict_dict, en_train_score_dict = \
        get_top_pages(en_bm25_model, NUM_PAGES_PREDICT, df_en_train, en_urls, result_dict_en_train, 'en', qid_dict_en_train)
    print ("Accuracy for en train: ", en_train_result)
    en_test_result, en_test_predict_dict, en_test_score_dict = \
        get_top_pages(en_bm25_model, NUM_PAGES_PREDICT, df_en_test, en_urls, result_dict_en_test, 'en', qid_dict_en_test)
    print ("Accuracy for en test: ", en_test_result)
    write_recall_to_csv(qid_dict_en_test, en_test_predict_dict, en_test_score_dict, OUTPUT_FILE_EN)

    tr_corpus, tr_urls, tr_bm25_model= build_bm25_model(os.path.join(DATA_DIR, "tr_corpus"), 'tr')
    print ("tr corpus processing finished")
    df_tr_train, qid_dict_tr_train, result_dict_tr_train = \
        get_data(os.path.join(DATA_DIR, "train", "train_tr.tsv"), NUM_PAGES_RESULT)
    print ("tr train processing finished")
    df_tr_test, qid_dict_tr_test, result_dict_tr_test = \
        get_data(os.path.join(DATA_DIR, "test", "test_tr.tsv"), NUM_PAGES_RESULT)
    print ("tr test processing finished")
    tr_train_result, tr_train_predict_dict, tr_train_score_dict = \
        get_top_pages(tr_bm25_model, NUM_PAGES_PREDICT, df_tr_train, tr_urls, result_dict_tr_train, 'tr', qid_dict_tr_train)
    print ("Accuracy for tr train: ", tr_train_result)
    tr_test_result, tr_test_predict_dict, tr_test_score_dict = \
        get_top_pages(tr_bm25_model, NUM_PAGES_PREDICT, df_tr_test, tr_urls, result_dict_tr_test, 'tr', qid_dict_tr_test)
    print ("Accuracy for tr test: ", tr_test_result)
    write_recall_to_csv(qid_dict_tr_test, tr_test_predict_dict, tr_test_score_dict, OUTPUT_FILE_TR)