from nltk.tokenize import word_tokenize


with open('vocab100.txt','r') as f:
    lines = f.readlines()
print lines

with open('vocab100Edited.txt','w') as g:
    for line in lines:
        word_tokenized = word_tokenize(line)
        n=0
        for word in word_tokenized:
            n=n+1
        if n==1:
            g.write(line)
