import csv

# changed the name of this file to binary learning output.
# this file format is same with format_checker.
GUESSED_OUTPUT_FILE = 'outputmanifold_mixup.tsv'

# original data file contains url's etc.
# that file will be used later in second multi labeling.

# label to match - if matched we pull this data from unchanged source data.
SELECTED_LABEL = 'no'

with open('data/CT22_english_1D_attentionworthy_dev_test_.tsv') as source_data:
    source_data_reader = csv.reader(source_data, delimiter='\t')
    with open(GUESSED_OUTPUT_FILE) as binary_labeled_data:
        binary_data_reader = csv.reader(binary_labeled_data, delimiter='\t')
        with open(f'guessed_{"positive" if SELECTED_LABEL == "yes" else "negative"}_test_data.tsv', 'w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter='\t')
            writer.writerow(next(source_data_reader))
            for source_row in source_data_reader:
                for binary_row in binary_data_reader:
                    if binary_row[1] == source_row[1] and binary_row[2] == SELECTED_LABEL:
                        writer.writerow(source_row)
                binary_labeled_data.seek(0)
