from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize


with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(f_contents)
    print(words)
