import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
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

input_file = "numerical_features.csv"
df = pd.read_csv(input_file, header = 0)
#df = (df - df.min())/(df.max() - df.min())   # Data Normalization
original_headers = list(df.columns.values)
numpy_matrix = df.as_matrix()
np.set_printoptions(linewidth = 200)
pd.set_option('expand_frame_repr',False)
pd.set_option('display.max_rows',365)
#print 'Numpy Matrix: '
#print df
'''
x = numpy_matrix[:,[3]]
print 'x-values: '
print(x)
print('\n')
y = numpy_matrix[:,[4]]
print 'y-values: '
print(y)
print('\n')
'''

c = 0
sum_of_coeff = 0
array_of_coeff = []
for item in numpy_matrix[:5]:
    coeff = np.corrcoef(numpy_matrix[:, c], numpy_matrix[:, 5])[0, 1]
    array_of_coeff.append(coeff)
    #print 'Correlation coeff of feature ',c,': ',coeff
    sum_of_coeff = sum_of_coeff + coeff
    c = c + 1

f = 0
array_of_weight = []
for coeff in array_of_coeff:
    weight = (coeff/sum_of_coeff)
    array_of_weight.append(weight)
    f = f + 1


a = 0
weighted_sum_of_features = []
for feature in numpy_matrix[:5]:
    feature_a = numpy_matrix[:,a]
    b = 0
    for item in feature_a:
        weighted_item = item * array_of_weight[a]
        #print weighted_item
        if a == 0:
            weighted_sum_of_features.append(weighted_item)
        else:
            weighted_sum_of_features[b] = weighted_sum_of_features[b] + weighted_item
            b = b + 1
    a = a + 1
'''
print 'Weighted sum of features: '
for item in weighted_sum_of_features:
    print item
'''


x = np.array(weighted_sum_of_features)
y = np.array(numpy_matrix[:,5])

x = x.reshape(-1,1)
y = y.reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(x,y)
#print 'Mean Squared Error: ',np.mean((regr.predict(x) - y)**2)
fig = plt.figure()
fig.suptitle('Training by linear regression', fontsize = 16)
plt.scatter(x, y,  color='black')
plt.plot(x,regr.predict(x), color='blue',linewidth=3)
plt.xlabel('Sum of weighted features')
plt.ylabel('Score')
#plt.show()


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


datadir = 'train_data/'
train_data = PlaintextCorpusReader(datadir, '.*')
all_contents = train_data.raw().strip()
all_text = preprocess(all_contents)
bag_of_words = [word for word, word_count in Counter(all_text).most_common(20)]

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


with open('test_data/test_1_4.txt', 'r') as f:
    f_contents = f.read()
preprocessed_text = preprocess(f_contents)
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

current_sum_of_features = array_of_weight[0]*BOW_Score + array_of_weight[1]*percent + array_of_weight[2]*v + array_of_weight[3]*no_of_words + array_of_weight[4]*sum
predicted_score = regr.predict(current_sum_of_features)
print 'Predicted score: ',round(predicted_score,0)