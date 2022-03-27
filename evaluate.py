import torch
from dataset import Dataset, INV_LABELS
import csv


def Evaluate(model, test_data, model_name=None):
    test = Dataset(test_data, model_name=model_name)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        current_entry = 0
        with open(f'output{model_name}.tsv', 'w', newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            writer.writerow(['topic', 'tweet_id', 'class_label', 'run_id'])
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                for ind in range(0, len(output.argmax(dim=1))):
                    writer.writerow(
                        [test.topic, test.tweet_ids[current_entry], INV_LABELS[int(output.argmax(dim=1)[ind])], test.model_name]
                    )
                    current_entry += 1

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')