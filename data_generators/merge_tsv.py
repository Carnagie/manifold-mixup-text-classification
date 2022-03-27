import csv

with open('data/2012_Sandy_Hurricane-ontopic_offtopic.tsv') as fr:
    reader = csv.reader(fr, delimiter='\t')
    with open('data/new_merged.tsv', 'w', newline='') as fw:
        writer = csv.writer(fw, delimiter='\t')
        next(reader)
        # writer.writerow(['topic', 'tweet_id', 'tweet_url', 'tweet_text', 'class_label'])
        for row in reader:
            class_label = 'yes_calls_for_action' if row[2] == 'on-topic' else 'no_not_interesting'
            writer.writerow(['COVID-19', row[0][:-1], 'no_url', row[1], class_label])
