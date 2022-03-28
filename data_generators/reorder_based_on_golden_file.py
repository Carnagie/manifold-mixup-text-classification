import csv

GOLDEN_FILE = 'data/CT22_english_1D_attentionworthy_dev_test.tsv'
GUESSED_FILE = 'outputmanifold_mixup_multi_stepped_final.tsv'
OUTPUT_FILE = 'reordered_multi_output_epoch_5.tsv'

with open(GOLDEN_FILE) as golden_file:
    golden_reader = csv.reader(golden_file, delimiter='\t')
    with open(GUESSED_FILE) as guessed_file:
        guessed_reader = csv.reader(guessed_file, delimiter='\t')
        with open(OUTPUT_FILE, 'w', newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            for golden_row in golden_reader:
                for guessed_row in guessed_reader:
                    if golden_row[1] == guessed_row[1]:
                        writer.writerow(guessed_row)
                guessed_file.seek(0)
