from nltk.tokenize import word_tokenize

with open('0001.txt','r') as f:
    f_contents = f.read()
    word_tokenized = word_tokenize(f_contents)
    print(word_tokenized)
    score = word_tokenized[-1]
    print score
    Score = int(score) + 1
    print Score
    print word_tokenized[:-1]