from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk import CFG
import nltk
from nltk import pos_tag

sent = "The boy saw a cat"
words = word_tokenize(sent)
tagged = pos_tag(words)
print(tagged)


grammar = CFG.fromstring("""
S -> NP VP
VP -> VBD NP
NP -> DT NN
""")

parser = nltk.ChartParser(grammar)
trees = parser.parse(tagged)
for tree in trees:
    print tree