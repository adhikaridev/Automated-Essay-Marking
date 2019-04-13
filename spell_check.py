from enchant import DictWithPWL
from enchant.checker import SpellChecker


my_dict = DictWithPWL("en_US","myDict.txt")
my_checker = SpellChecker(my_dict)
with open('test_copy.txt','r') as f:
    f_contents = f.read().decode("utf-8-sig").encode("utf-8") #decode the contents to unicode and encode to utf-8
    my_checker.set_text(f_contents)
    e =0
    for error in my_checker:
        print "ERROR:", error.word
        e = e+1
    print('No. of errors: ',e)
'''
import enchant
import wx
from enchant.checker import SpellChecker
from enchant.checker.wxSpellCheckerDialog import wxSpellCheckerDialog
from enchant.checker.CmdLineChecker import CmdLineChecker


a = "Cats are animalss. " \
    "They are violenttt."
chkr = enchant.checker.SpellChecker("en_US")
chkr.set_text(a)
for err in chkr:
    print err.word
    sug = err.suggest()[0]
    err.replace(sug)

c = chkr.get_text()#returns corrected text
print c
'''