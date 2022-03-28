import csv

with open('outputmanifold_mixup_binary.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    with open(f'outputmanifold_mixup_binary_no_not_interesting.tsv', 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for row in csv_reader:
            if row[2] == 'no':
                row[2] = 'no_not_interesting'
                writer.writerow(row)
