import re

with open('test.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8")   #decode the contents to unicode and encode to utf-8
    print len(re.findall(r'\w+', f_contents))

