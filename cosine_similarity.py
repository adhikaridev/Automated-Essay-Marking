import glob
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from numpy import array,dot
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

path = '/home/sudo-this/PycharmProjects/Automated Essay Marking/top_scored_essay/*.txt'
files = glob.glob(path)

reference_collection = []
for filename in sorted(files):
    with open(filename, 'r') as f:
        f_contents = f.read()
    preprocessed_text = preprocess(f_contents)
    reference_text = preprocessed_text[:-1]
    reference_collection.append(reference_text)

with open('0017.txt','r') as f:
    current_contents = f.read()
    current_contents = preprocess(current_contents)
    current_text = current_contents[:-1]
i = 1
sum = 0
print 'Cosine similarity: '
for text in reference_collection:
    reference_vector = []
    current_vector = []
    combined_set = text + current_text
    sorted_unique = sorted(list(set(combined_set)))
    for sorted_item in sorted_unique:
        for word, word_count in Counter(text).most_common(10000):
            if sorted_item == word:
                reference_vector.append(word_count)
        if sorted_item not in text:
            reference_vector.append(0)

        for word, word_count in Counter(current_text).most_common(10000):
            if sorted_item == word:
                current_vector.append(word_count)
        if sorted_item not in current_text:
            current_vector.append(0)
    reference_vector = array(reference_vector)
    current_vector = array(current_vector)

    c = dot(reference_vector, current_vector) / norm(reference_vector) / norm(current_vector)
    print 'Similarity with reference essay',i,':',c*100
    sum = sum + c
    i += 1
avg = sum*100/5
print '\nAverage similarity: ',avg
