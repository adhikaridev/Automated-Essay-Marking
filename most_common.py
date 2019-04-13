'''
import collections

words = ['all','you','yes','all']
counter = collections.Counter(words)
print(counter.most_common(1))
print most_common_words
'''

from collections import Counter
words = ['all','you','yes','all']
most_common_words = [word for word, word_count in Counter(words).most_common(1)]
print most_common_words
