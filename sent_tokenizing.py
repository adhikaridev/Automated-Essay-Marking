from nltk.tokenize import sent_tokenize, word_tokenize

with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    sent_tokenized = sent_tokenize(f_contents)
    print(sent_tokenized)