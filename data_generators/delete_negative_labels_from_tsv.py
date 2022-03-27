import csv

with open('data/2012_Sandy_Hurricane-ontopic_offtopic.tsv') as fr:
    reader = csv.reader(fr, delimiter='\t')
    with open('data/new_merged.tsv', 'w', newline='') as fw:
        writer = csv.writer(fw, delimiter='\t')
        next(reader)
        writer.writerow(['topic', 'tweet_id', 'tweet_url', 'tweet_text', 'class_label'])
        for row in reader:
            if row[-1] == 'no_not_interesting':
                continue
            writer.writerow(row)
