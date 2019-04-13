import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter

corpusdir = 'train_corpus/'
newcorpus = PlaintextCorpusReader(corpusdir, '.*')

#print newcorpus.raw().strip()

all_contents = newcorpus.raw().strip()

#print(all_contents)

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(all_contents)
stopwords = set(stopwords.words("english"))
filtered_contents = [w for w in words if not w in stopwords]
lowered_words = []
for w in filtered_contents:
    lowered_words.append(w.lower())
#print lowered_words

most_common_words = [word for word,word_count in Counter(lowered_words).most_common(20)]
print most_common_words

