import spacy
from nltk import Tree
en_nlp = spacy.load('en')
doc = en_nlp(u"Computers are very important in our time.")
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_
[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]