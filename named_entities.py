from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk


def entities(text):
    return ne_chunk(
        pos_tag(
            word_tokenize(text)
        )
    )
with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
tree = entities(f_contents)
tree.pprint()
tree.draw()