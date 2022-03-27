import csv
import random
from random import randrange

with open('data/CT22_dutch_1D_attentionworthy_dev_test.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    with open(f'data/CT22_dutch_1D_attentionworthy_dev_test_reduced.tsv', 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            if row[-1] == 'no_not_interesting':
                choice = random.randrange(80)
                if choice == 0:
                    writer.writerow(row)
            elif row[-1] == 'harmful':
                choice = random.randrange(5)
                if choice == 0:
                    writer.writerow(row)
            elif row[-1] == 'yes_blame_authorities':
                choice = random.randrange(8)
                if choice == 0:
                    writer.writerow(row)
            else:
                writer.writerow(row)
