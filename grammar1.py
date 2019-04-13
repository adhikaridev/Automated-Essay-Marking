from nltk import CFG
import nltk
from nltk import pos_tag


grammar = CFG.fromstring("""
S -> NP VP
VP -> V NP | V NP PP
PP -> P NP
V -> "saw" | "ate" | "walked"
NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
Det -> "a" | "an" | "the" | "my"
N -> "man" | "dog" | "cat" | "telescope" | "park"
P -> "in" | "on" | "by" | "with"
""")

sent = "the dog saw a man in the park".split()
parser = nltk.ChartParser(grammar)
trees = parser.parse(sent)
for tree in trees:
    print tree