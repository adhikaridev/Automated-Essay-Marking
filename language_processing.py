import os
import re
import glob
import csv
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from enchant import DictWithPWL
from enchant.checker import SpellChecker
from numpy import array, dot
from numpy.linalg import norm

stopwords = set(stopwords.words("english"))


def preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    punct_removed = tokenizer.tokenize(text)
    lowered_words = []
    for w in punct_removed:
        lowered_words.append(w.lower())
    filtered_contents = [w for w in lowered_words if not w in stopwords]
    lemmatized_contents = []
    lemmatizer = WordNetLemmatizer()
    for w in filtered_contents:
        lemmatized_contents.append(lemmatizer.lemmatize(w))
    return lemmatized_contents


def BOW_score(text):
    BOW_contents_individual = [w for w in text if w in bag_of_words]

    refined_BOW_individual = []
    n = 0
    for word in BOW_contents_individual:
        if word not in refined_BOW_individual:
            refined_BOW_individual.append(word)
            n = n + 1
    print '    Bag of words matches: '
    print '\t\t\t', refined_BOW_individual
    return n


def spelling(text):
    my_dict = DictWithPWL("en_US", "myDict.txt")
    my_checker = SpellChecker(my_dict)
    my_checker.set_text(text)
    e = 0
    print '    Spelling errors: '
    for error in my_checker:
        print "              ", error.word
        e = e + 1
    return e


def vocab_check(f_contents):
    text = word_tokenize(f_contents)
    used_vocab = [w for w in text if w in v_words]

    refined_used_vocab = []
    n = 0
    for word in used_vocab:
        if word not in refined_used_vocab:
            refined_used_vocab.append(word)
            n = n + 1
    print '    Used Vocab words: '
    print '\t\t\t', refined_used_vocab
    print '    No. of vocab words used: ', n, '\n'
    return n

'''
datadir = 'train_data/'
train_data = PlaintextCorpusReader(datadir, '.*')
all_contents = train_da                                                     ta.raw().strip()
all_text = preprocess(all_contents)
'''
with open('bag_of_words.txt', 'r') as b:
    b_contents = b.read()
    bag_of_words = word_tokenize(b_contents)
#bag_of_words = [word for word, word_count in Counter(all_text).most_common(20)]
print('\n Bag of Words:')
print bag_of_words
print('\n')

path1 = '/home/sudo-this/PycharmProjects/Automated Essay Marking/top_scored_essay/*.txt'
files = glob.glob(path1)
reference_collection = []
for filename in sorted(files):
    with open(filename, 'r') as f:
        f_contents = f.read()
    preprocessed_text = preprocess(f_contents)
    reference_text = preprocessed_text[:-1]
    reference_collection.append(reference_text)

# reading high frequency vocab words
with open('vocab.txt', 'r') as v:
    v_contents = v.read()
    v_words = word_tokenize(v_contents)

path = '/home/sudo-this/PycharmProjects/Automated Essay Marking/train_data/*.txt'
files = glob.glob(path)
file_id = 0
with open('numerical_features.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['BOW_Score', '% of spelling errors', 'Vocab','No. of words', 'Content Similarity (in %)',
                     'Human_rated_score'])
    for filename in sorted(files):
        with open(filename, 'r') as f:
            f_contents = f.read().decode("utf-8-sig")
        file_id = file_id + 1
        print 'Essay ', file_id, ':'
        preprocessed_text = preprocess(f_contents)
        string_score = preprocessed_text[-1]
        int_score = int(string_score)
        human_rated_score = int_score
        print '    Human-rated score: ', human_rated_score
        preprocessed_text = preprocessed_text[:-1]
        BOW_Score = BOW_score(preprocessed_text)
        print '    BOW score: ', BOW_Score
        print('\n')

        e = spelling(f_contents)
        print '    No. of spelling errors: ', e, '\n'
        no_of_words = float(len(re.findall(r'\w+', f_contents)))
        print '    Total no. of words: ', no_of_words
        percent = (e / no_of_words) * 100
        print '    Percentage of spelling errors: ', percent, '\n'

        v = vocab_check(f_contents)

        sum = 0
        for text in reference_collection:
            reference_vector = []
            current_vector = []
            combined_set = text + preprocessed_text
            sorted_unique = sorted(list(set(combined_set)))
            for sorted_item in sorted_unique:
                for word, word_count in Counter(text).most_common(10000):
                    if sorted_item == word:
                        reference_vector.append(word_count)
                if sorted_item not in text:
                    reference_vector.append(0)

                for word, word_count in Counter(preprocessed_text).most_common(10000):
                    if sorted_item == word:
                        current_vector.append(word_count)
                if sorted_item not in preprocessed_text:
                    current_vector.append(0)
            reference_vector = array(reference_vector)
            current_vector = array(current_vector)

            c = dot(reference_vector, current_vector) / norm(reference_vector) / norm(current_vector)
            sum = sum + c
        similarity = int(sum * 100 / 5)
        print '    Content Similarity: ', similarity, '%'

        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([BOW_Score, percent, v, no_of_words, similarity, human_rated_score])