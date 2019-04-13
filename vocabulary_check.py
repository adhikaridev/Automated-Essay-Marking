from nltk import word_tokenize


with open('vocab.txt','r') as v:
    v_contents = v.read()
    v_words = word_tokenize(v_contents)

with open('0001.txt','r') as f:
    f_contents = f.read()
    text = word_tokenize(f_contents)

used_vocab = [w for w in text if w in v_words]

refined_used_vocab = []
n = 0
for word in used_vocab:
    if word not in refined_used_vocab:
        refined_used_vocab.append(word)
        n = n + 1
print 'Used Vocab words: '
print '\t\t\t',refined_used_vocab
print 'No. of vocab words used: ',n