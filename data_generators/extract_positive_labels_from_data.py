import csv

with open('data/CT22_english_1D_attentionworthy_dev_test.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    with open(f'data/CT22_english_1D_attentionworthy_dev_test_positives.tsv', 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            if row[-1] != 'no_not_interesting':
                writer.writerow(row)
