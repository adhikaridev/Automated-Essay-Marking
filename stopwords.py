from nltk import word_tokenize
from nltk.corpus import stopwords


with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    word_tokenized = word_tokenize(f_contents)
    stopwords = set(stopwords.words("english"))
    filtered_sentence = [w for w in word_tokenized if not w in stopwords]
    print(filtered_sentence)