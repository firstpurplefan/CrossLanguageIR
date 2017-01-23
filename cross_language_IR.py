import nltk
import string
import collections
import math
from nltk.tokenize import word_tokenize
from six import iteritems
from six.moves import xrange

punctuation = set(string.punctuation)
english_stop_words = set(nltk.corpus.stopwords.words('english'))
german_stop_words = set(nltk.corpus.stopwords.words('german'))

porter_stemmer = nltk.stem.porter.PorterStemmer()

def tokenize(line, tokenizer = word_tokenize):
    utf_line = line.decode('utf-8').lower()
    return [token.encode('ascii', 'backslashreplace') for token in tokenizer(utf_line)]

def preprocess(doc):
    terms = []
    for token in tokenize(doc):
        if token not in punctuation and token not in english_stop_words:
            terms.append(token)
    return terms

def preprocess_de(doc):
    terms = []
    for token in tokenize(doc):
        if token not in punctuation and token not in german_stop_words:
            terms.append(token)
    return terms

def get_doc_list(doc_path):
    querys = []
    with open(doc_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            querys.append(preprocess(line))
    return querys

import pickle
def load_object(file_path):
    with open(file_path, 'rb') as f:
        object_instance = pickle.load(f)
    return object_instance

def saving_object(file_path, object_instance):
    with open(file_path, 'wb') as f:
        pickle.dump(object_instance, f, pickle.HIGHEST_PROTOCOL)

class PostingListConstructor(object):
    """docstring for PostingListConstructor"""
    def __init__(self, docs_path):
        super(PostingListConstructor, self).__init__()
        self.docs_path = docs_path
        self.posting_lists = {} # {term:(term_count, posting_list)}
        self.num_docs_with_term = {}
        self.len_docs = {} # {doc_id: _, doc_length: _}
        self.num_docs = 0
        self.total_doc_length = 0
        self.doc_wordcounters = []
        self.original_lines = []
        with open(docs_path, 'r') as f:
            lines = f.readlines()
            check_next = 0
            words = []
            for i in range(len(lines)):
                self.num_docs += 1
                doc_id, text = lines[i].split('\t')
                terms = []
                for token in tokenize(text):
                    if token not in punctuation and token not in english_stop_words:
                        terms.append(token)
                        words.append(token)
                        if (self.posting_lists.get(token, []) == []):
                            self.posting_lists[token] = self.posting_lists.get(token, []) + [i]
                        else:
                            if self.posting_lists[token][-1] != i:
                                self.posting_lists[token] = self.posting_lists.get(token, []) + [i]
                        if (self.num_docs_with_term.get(token, 0) == 0):
                            self.num_docs_with_term[token] = self.num_docs_with_term.get(token, 0) + 1
                        else:
                            if self.posting_lists[token][-1] != i:
                                self.num_docs_with_term[token] = self.num_docs_with_term.get(token, 0) + 1
                self.doc_wordcounters.append(collections.Counter(terms))
                self.original_lines.append(terms)
                self.len_docs[i] = len(terms)
                self.total_doc_length += len(terms)
        self.terms = list(set(words))

    def generate_td_list(self):
        terms = self.terms
        td_idf_list = []
        for term in terms:
            pointers = self.posting_lists[term]
            doc_freq = self.num_docs_with_term[term]
            idf = math.log(float(self.num_docs)/float(doc_freq))
            td_sub_list = []
            idf_list = []
            for pointer in pointers:
                td_sub_list.append((pointer, self.doc_wordcounters[pointer][term]*idf))
            td_idf_list.append((term, td_sub_list))
        td_idf = dict(td_idf_list)
        return td_idf

posting_list_generator = PostingListConstructor('./dev.docs')
saving_object('./dev.constructor', posting_list_generator)

class BM25(object):

    def __init__(self, corpus_constructor, k1 = 1.5, b= 0.75, k3=0.1):
        self.corpus_size = corpus_constructor.num_docs
        self.avgdl = float(sum(corpus_constructor.len_docs.values())) / self.corpus_size
        self.posting_list_generator = corpus_constructor
        self.f = []
        self.df = {}
        self.idf = {}
        self.PARAM_K1 = k1
        self.PARAM_B = b
        self.PARAM_K3 = k3
    def get_sent_score(self, querys):
        idf = []
        document_index = []
        document_scores = {}
        PARAM_K1 = self.PARAM_K1
        PARAM_B  = self.PARAM_B
        PARAM_K3 = self.PARAM_K3
        for query in querys:
            pointers = self.posting_list_generator.posting_lists.get(query, 0)
            if pointers != 0:
                document_index.extend(pointers)
        document_index = list(set(document_index))
        for i in document_index:
            scoredq = 0
            for query in querys:
                score = 0
                n = self.posting_list_generator.num_docs
                nqi = self.posting_list_generator.num_docs_with_term.get(query, 0)
                idfi = math.log((n-nqi+0.5) /(nqi+0.5))
                fqid = self.posting_list_generator.doc_wordcounters[i].get(query, 0)
                absd = self.posting_list_generator.len_docs[i]
                tfi = (fqid*(PARAM_K1+1))/(fqid+PARAM_K1*(1-PARAM_B+PARAM_B*absd/self.avgdl))
                qtf = ((PARAM_K3+1)*fqid)/(PARAM_K3+fqid)
                scorei = idfi*tfi*qtf
                scoredq += scorei
            document_scores[i] = scoredq
        return document_scores

def get_sent_querys_dict(query_path, bm25model):
    querylist = []
    querys = get_doc_list(query_path)
    for query in querys:
        querylist.append(bm25model.get_sent_score(query))
    return querylist

import operator

bm25model = BM25(posting_list_generator)
querys = get_doc_list('./test_query')
querytest = get_sent_querys_dict('./test_query', bm25model)

for i in range(len(querytest)):
    max_index = max(querytest[i].iteritems(), key=operator.itemgetter(1))[0]
    print querys[i], max_index, posting_list_generator.original_lines[max_index]

import nltk
import string
import collections
import math
from nltk.tokenize import word_tokenize
from six import iteritems
from six.moves import xrange

punctuation = set(string.punctuation)
english_stop_words = set(nltk.corpus.stopwords.words('english'))
german_stop_words = set(nltk.corpus.stopwords.words('german'))

porter_stemmer = nltk.stem.porter.PorterStemmer()

posting_lists = {}
token = 'abc'
posting_lists[token] = posting_list.get(token, []).append[i]

def tokenize(line, tokenizer = word_tokenize):
    utf_line = line.decode('utf-8').lower()
    return [token.encode('ascii', 'backslashreplace') for token in tokenizer(utf_line)]

def preprocess(doc):
    terms = []
    for token in tokenize(doc):
        if token not in punctuation and token not in english_stop_words:
            terms.append(token)
    return terms

def preprocess_de(doc):
    terms = []
    for token in tokenize(doc):
        if token not in punctuation and token not in german_stop_words:
            terms.append(token)
    return terms

def get_doc_list(doc_path):
    querys = []
    with open(doc_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            querys.append(preprocess(line))
    return querys

#this trigram with backoff smoothing
#with small bugs
from nltk.util import ngrams
PARAM_K = 0
def get_en_list(doc_path):
    querys = []
    with open(doc_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            querys.append(preprocess(line))
    return querys

def get_de_list(doc_path):
    querys = []
    with open(doc_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            querys.append(preprocess_de(line))
    return querys

#original list of list of token, after punctuation and stop word removal
origianl_docs = get_en_list('./bitext-large.en')

from collections import defaultdict

def check_for_unk_train(word,unigram_counts):
    if word in unigram_counts:
        return word
    else:
        unigram_counts[word] = 0
        return "UNK"

def convert_sentence_train(sentence,unigram_counts):
    return ["<s>"] + [check_for_unk_train(token.lower(),unigram_counts) for token in sentence] + ["</s>"]

def convert_tri_sentence_train(sentence,unigram_counts):
    return ["<s>","<s>"] + [check_for_unk_train(token.lower(),unigram_counts) for token in sentence] + ["</s>", "</s>"]

def get_bigram_counts(sentences):
    bigram_counts = defaultdict(dict)
    unigram_counts = {}
    for sentence in sentences:
        sentence = convert_sentence_train(sentence, unigram_counts)
        for i in range(len(sentence) - 1):
            bigram_counts[sentence[i]][sentence[i+1]] = bigram_counts[sentence[i]].get(sentence[i+1],0) + 1
            unigram_counts[sentence[i]] = unigram_counts.get(sentence[i],0) + 1
    token_count = float(sum(unigram_counts.values()))
    unigram_counts["</s>"] = unigram_counts["<s>"]
    return unigram_counts, bigram_counts, token_count

def get_trigram_counts(sentences):
    trigram_counts = defaultdict(dict)
    unigram_counts = {}
    for sentence in sentences:
        sentence = convert_tri_sentence_train(sentence, unigram_counts)
        for i in range(len(sentence) - 2):
            trigram_counts[(sentence[i], sentence[i+1])][sentence[i+2]] = trigram_counts[(sentence[i], sentence[i+1])].get(sentence[i+2],0) + 1
            unigram_counts[sentence[i]] = unigram_counts.get(sentence[i],0) + 1
    token_count = float(sum(unigram_counts.values()))
    unigram_counts["</s>"] = unigram_counts["<s>"]
    return unigram_counts, trigram_counts, token_count

from numpy.random import choice 

def generate_sentence(bigram_counts):
    sentence = ["<s>"]
    while sentence[-1] != "</s>":
        prev_token_counts = bigram_counts[sentence[-1]]
        bigram_probabilities = []
        total_counts = float(sum(prev_token_counts.values()))
        for token in prev_token_counts:
            bigram_probabilities.append(prev_token_counts[token]/total_counts)
        sentence.append(choice(prev_token_counts.keys(), p=bigram_probabilities))
    return " ".join([sentence[1].title()] + sentence[2:-1]).replace(" ,",",").replace(" .", ".")

docs_unigrams, docs_trigrams, docs_token_count = get_trigram_counts(origianl_docs)
docs_unigrams1, docs_bigrams, docs_token_count1 = get_bigram_counts(origianl_docs)
docs_uni_value_counts = collections.Counter(docs_unigrams1.values())

import math

def get_log_prob_interp(sentence,i, unigram_counts,bigram_counts, trigram_counts, token_count, bigram_lambda, trigram_lambda):
    tri_full_hit = trigram_counts.get((sentence[i-2],sentence[i-1]), 0)
    if tri_full_hit != 0:
        tri_full_hit = tri_full_hit.get(sentence[i], 0)
    return math.log(trigram_lambda*tri_full_hit/float(bigram_counts[sentence[i-2]].get(sentence[i-1], 1)) + 
        (1 - trigram_lambda) *
        (bigram_lambda*bigram_counts[sentence[i-1]].get(sentence[i],0)/float(unigram_counts[sentence[i-1]]) + 
                    (1 - bigram_lambda)*unigram_counts[sentence[i]]/token_count))

def check_for_unk_test(word,unigram_counts):
    if word in unigram_counts and unigram_counts[word] > 0:
        return word
    else:
        return "UNK"

def convert_sentence_test(sentence,unigram_counts):
    return ["<s>", "<s>"] + [check_for_unk_test(word.lower(),unigram_counts) for word in sentence] + ["</s>", "</s>"]

def convert_sentence_uni_test(sentence,unigram_counts):
    return ["<s>"] + [check_for_unk_test(word.lower(),unigram_counts) for word in sentence] + ["</s>"]

def get_sent_log_prob_interp(sentence, unigram_counts, bigram_counts, trigram_counts, token_count, bigram_lambda, trigram_lambda):
    sentence = convert_sentence_test(sentence, bigram_counts)
    return sum([get_log_prob_interp(sentence,i, unigram_counts,bigram_counts, trigram_counts, token_count, bigram_lambda, trigram_lambda) for i in range(2,len(sentence))])

from decimal import *

def get_log_uni_prob_gt(token, unigram_counts, value_counts, token_count, k):
    c = unigram_counts.get(token, 0)
    nc = value_counts.get(c, 1)
    nc1 = value_counts.get(c+1, 1)
    if c != 0:
        c_star = c
    else:
        c_star = (c+1)*nc1/nc
    p = Decimal(c_star)/Decimal(token_count)
    if k == 1:
        return float(c_star)/token_count
    else:
        return p.ln()

def get_sent_uni_log_prob_gt(sentence, unigram_counts, value_counts, token_count):
    sentence = convert_sentence_uni_test(sentence, unigram_counts)
    return sum([get_log_uni_prob_gt(sentence[i], unigram_counts, value_counts, token_count, 2) for i in range(0,len(sentence))])

sentence = "president".split()
print convert_sentence_test(sentence, docs_bigrams)
print get_sent_log_prob_interp(sentence, docs_unigrams1, docs_bigrams, docs_trigrams, docs_token_count1, 0.95, 0.95)
print get_sent_uni_log_prob_gt(sentence, docs_unigrams1, docs_uni_value_counts, docs_token_count1)

def calculate_perplexity(sentences,unigram_counts,bigram_counts, trigram_counts, token_count, smoothing_function, parameter1, parameter2):
    total_log_prob = 0
    test_token_count = 0
    for sentence in sentences:
        test_token_count += len(sentence) + 2 # have to consider the end token
        total_log_prob += smoothing_function(sentence,unigram_counts,bigram_counts, trigram_counts, token_count, parameter1, parameter2)
    return math.exp(-total_log_prob/test_token_count)

test_set = get_en_list('./newstest2013.en')

print "interpolation"
for bigram_lambda in [0.99,0.95,0.75,0.5,0.25,0.001]:
    for trigram_lambda in [0.99,0.95,0.75,0.5,0.25,0.001]:
        print bigram_lambda, trigram_lambda
        print calculate_perplexity(test_set,docs_unigrams,docs_bigrams,docs_trigrams, docs_token_count,get_sent_log_prob_interp,bigram_lambda, trigram_lambda)   

from nltk.translate import IBMModel1
from nltk.translate import AlignedSent, Alignment

def get_en_list(doc_path):
    querys = []
    with open(doc_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            querys.append(preprocess(line))
    return querys

def get_de_list(doc_path):
    querys = []
    with open(doc_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            querys.append(preprocess_de(line))
    return querys


en_text = get_en_list('./bitext-small.en')
de_text = get_de_list('./bitext-small.de')

all_en_words_set = set([item for sublist in en_text for item in sublist])
all_de_words_set = set([item for sublist in de_text for item in sublist])

all_en_words = list(all_en_words_set)
all_de_words = list(all_de_words_set)

#english to german
bitext_en_de = zip(en_text, de_text)
bt_en_de = [AlignedSent(E,F) for E,F in bitext_en_de]
imb1_en_de = IBMModel1(bt_en_de, 5)
tm_en_de = imb1_en_de.translation_table

#english to german
bitext_de_en = zip(de_text, en_text)
bt_de_en = [AlignedSent(E,F) for E,F in bitext_de_en]
imb1_de_en = IBMModel1(bt_de_en, 5)
tm_de_en = imb1_de_en.translation_table

print 'finished'

print tm_en_de['area']
print tm_de_en['haus']

from decimal import *

def get_log_uni_prob_gt(token, unigram_counts, value_counts, token_count, k):
    c = unigram_counts.get(token, 0)
    nc = value_counts.get(c, 1)
    nc1 = value_counts.get(c+1, 1)
    if c != 0:
        c_star = c
    else:
        c_star = (c+1)*nc1/nc
    p = Decimal(c_star)/Decimal(token_count)
    if k == 1:
        return float(c_star)/token_count
    else:
        return p.ln()

def get_sent_uni_log_prob_gt(sentence, unigram_counts, value_counts, token_count):
    sentence = convert_sentence_test(sentence, unigram_counts)
    return sum([get_log_uni_prob_gt(sentence[i], unigram_counts, value_counts, token_count, 5) for i in range(0,len(sentence))])

def calculate_uni_perplexity(sentences,unigram_counts,token_count, smoothing_function):
    value_counts = collections.Counter(unigram_counts.values())
    total_log_prob = 0
    test_token_count = 0
    for sentence in sentences:
        test_token_count += len(sentence)+1 # have to consider the end token
        total_log_prob += smoothing_function(sentence,unigram_counts, value_counts, token_count)
    return math.exp(-total_log_prob/test_token_count)

print "Unigram Perplexity with Good Turing Smoothing: "
print calculate_uni_perplexity(en_text,docs_unigrams1,docs_token_count1, get_sent_uni_log_prob_gt)

de_query_list = get_de_list("./test.queries")

def decode(query, translation_model, unigram_counts, bigram_counts, trigram_counts, token_count, n):
    value_counts = collections.Counter(unigram_counts.values())
    final_result = []
    for de_token in query:
        string = ""
        prob = -999999
     
        for token, value in translation_model[de_token].iteritems():
            if((token != 'None') and token not in english_stop_words):
                uni_log_prob = get_log_uni_prob_gt(token, unigram_counts, value_counts, token_count, 1)
                prob_tmp = value * uni_log_prob
                if prob_tmp > prob:
                    prob = prob_tmp
                    string = token
        final_result.append(string)
    return final_result

query = de_query_list[0]
query = ['verliert']
print decode(query, tm_de_en, docs_unigrams1, docs_bigrams, docs_trigrams, docs_token_count1, 1)