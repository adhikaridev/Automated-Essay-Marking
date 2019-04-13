from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import *
from nltk.stem.snowball import EnglishStemmer

with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    words = word_tokenize(f_contents)
    stemmer = PorterStemmer()
    #for w in words:
     #   print(stemmer.stem(w))
    print stemmer.stem('having')

    stemmer2 = SnowballStemmer('english')
    print stemmer2.stem('distribution')

    stemmer3 = EnglishStemmer()
    print stemmer3.stem('require')