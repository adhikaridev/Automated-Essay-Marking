from nltk.tokenize import word_tokenize

with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    word_tokenized = word_tokenize(f_contents)
    #print(f_contents)
    print(word_tokenized)
