import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk.stem import WordNetLemmatizer

corpusdir = 'train_data/'
newcorpus = PlaintextCorpusReader(corpusdir, '.*')

#print newcorpus.raw().strip()

all_contents = newcorpus.raw().strip()

#print(all_contents)

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(all_contents)
lowered_words = []
for w in words:
    lowered_words.append(w.lower())
stopwords = set(stopwords.words("english"))
filtered_contents = [w for w in lowered_words if not w in stopwords]
lemmatized_contents = []
lemmatizer = WordNetLemmatizer()
for w in filtered_contents:
    lemmatized_contents.append(lemmatizer.lemmatize(w))

#most_common_words = [word for word,word_count in Counter(lemmatized_contents).most_common(20)]
#bag_of_words = most_common_words
#print bag_of_words

print Counter(lemmatized_contents).most_common(100)