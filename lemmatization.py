from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    words = word_tokenize(f_contents)
    lemmatizer = WordNetLemmatizer()
    for w in words:
        print(lemmatizer.lemmatize(w))