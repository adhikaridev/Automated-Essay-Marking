from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    words = word_tokenize(f_contents)
    lowered_words = []
    for w in words:
        lowered_words.append(w.lower())
    lemmatizer = WordNetLemmatizer()
    for w in lowered_words:
        print(lemmatizer.lemmatize(w))