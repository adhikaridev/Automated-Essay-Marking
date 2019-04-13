import glob

path = '/home/sudo-this/PycharmProjects/Automated Essay Marking/train_data/*.txt'
files = glob.glob(path)
for filename in files:
    with open(filename, 'r') as f:
        f_contents = f.read()
        print f_contents