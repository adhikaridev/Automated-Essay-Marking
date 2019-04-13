import csv

a=2
b=5
with open('eggs.csv','wb') as csv_file:
    writer = csv.writer(csv_file,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['spam','lovely spam','wonderful spam'])
    writer.writerow(['spam']*5)
    writer.writerow([a])
