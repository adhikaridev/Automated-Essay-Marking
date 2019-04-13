from nltk.corpus import wordnet

syns = wordnet.synsets("program")

#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

#definition
print(syns[0].definition())  #gives the definition of plan

#examples
print(syns[0].examples())


#synonyms and antonyms

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))


#semantic similarity

w1 = wordnet.synset("ship.n.01")    #means ship with noun and its 1st meaning
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))