#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import re
import numpy

test_file = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test.data'

train_file = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_train.data'

vocab_file = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin.vocab'

test_file_sparse = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test_sparse.data'

train_file_sparse = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_train_sparse.data'



# Extract word in a sentence to form a list without stop words
def word_extraction(sentence):
    # set(stopwords.words('english'))
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text

def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    return vocab
    print("Word List for Document \n{0} \n".format(vocab));

#transfer the words list to sparse format
def words2sparse(vocab, words):
    word_sparse = {}
    for w in words:
        for i,word in enumerate(vocab):
            if word == w:
                if i in word_sparse:
                    word_sparse[i] += 1
                else:
                    word_sparse[i] = 1 
    return word_sparse

#The experiment for having get all the data
#for sentence in sentences: 
def write_to_sparse(test_query_sentences,test_title_sentences, file_path, test_query_label, vocab):
# generating the sparse format data for the training and testing
   f = open(file_path, 'w')
   data_size = len(test_query_sentences)
   feature_dim = len(vocab)

   f.write("%d,%d\n" % (data_size, feature_dim))

   for j in range(data_size):
        query_sentence = test_query_sentences[j]
        title_sentence = test_title_sentences[j]

        query_words = word_extraction(query_sentence)
        title_words = word_extraction(title_sentence)

        #print("{0},".format(test_query_label[j].replace('\n',''))),
        f.write("%s, " % test_query_label[j].replace('\n',''))
        #print("{0},".format(test_query_label[j].replace('\n','')), f),
    
        query_words_sparse = words2sparse(vocab, query_words)
        title_words_sparse = words2sparse(vocab, title_words)

        for w in query_words_sparse:
            #print("{0}:{1}".format(w, query_words_sparse[w])),
            f.write("%s:%s " % (w, query_words_sparse[w]))
            #print("{0}:{1}".format(w, query_words_sparse[w]), f),

        #print(","),
        #print(",", f)
        f.write(",")
        for w in title_words_sparse:
            #print("{0}:{1}".format(w, title_words_sparse[w])),
            f.write("%s:%s " % (w, title_words_sparse[w]))
            #print("{0}:{1}".format(w, title_words_sparse[w]), f),
        #print("\n")
        f.write("\n")
allsentences = []
test_query_sentences = []
test_title_sentences = []
test_query_label = []

train_title_sentences = []
train_query_sentences = []
train_query_label = []

with open(test_file) as tf:
    for line in tf:
        items = line.split(',')

        test_query_sentences.append(items[0])
        test_title_sentences.append(items[1])
        test_query_label.append(items[2])
with open (train_file) as tf:
    for line in tf:
        items = line.split(',')
        # Data cleaning for some wired data points
        if (len(items) > 2):
            train_query_sentences.append(items[0])
            train_title_sentences.append(items[1])
            train_query_label.append(items[2])

allsentences.extend(test_query_sentences)
allsentences.extend(test_title_sentences)
allsentences.extend(train_query_sentences)
allsentences.extend(train_title_sentences)


sentences = ["Joe waited for the train train", "The train was late", "Mary and Samantha took the bus",
        "I looked for Mary and Samantha at the bus station",
        "Mary and Samantha arrived at the bus station early but waited until noon for the bus"]


vocab = generate_bow(allsentences)

with open (vocab_file, 'w') as vf:
    for w in vocab:
        vf.write("%s\n" % (w))

write_to_sparse(test_query_sentences,test_title_sentences, test_file_sparse, test_query_label, vocab)
write_to_sparse(train_query_sentences,train_title_sentences, train_file_sparse, train_query_label, vocab)

# generating the sparse format data for the training and testing
#with open(test_file_sparse, 'w')as tf:

#with open (train_file_sparse, 'w') as tf:

