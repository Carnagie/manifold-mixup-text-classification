import csv
import random
from random import randrange

with open('data/CT22_english_1D_attentionworthy_train.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    with open(f'data/CT22_english_1D_attentionworthy_train_reduced_1_to_5.tsv', 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            if row[-1] == 'no_not_interesting':
                choice = random.randrange(5)
                if choice == 0:
                    writer.writerow(row)
            else:
                writer.writerow(row)
