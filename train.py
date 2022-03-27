import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from dataset import Dataset
from data_augmenters import get_sentence_mix_up_dict_and_label
from data_augmenters import get_manifold_mix_up_dict_and_label
from data_augmenters import get_word_mix_up_dict_and_label


def Train(model, train_data, val_data, learning_rate, epochs, mix_up_type=None, model_name=None):
    train, val = Dataset(train_data, model_name), Dataset(val_data, model_name)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            if mix_up_type == 'word_mixup':
                mix_up_dict, train_label = get_word_mix_up_dict_and_label(input_id, train_label)
            elif mix_up_type == 'sentence_mixup':
                mix_up_dict, train_label = get_sentence_mix_up_dict_and_label(input_id, train_label)
            elif mix_up_type == 'manifold_mixup':
                mix_up_dict, train_label = get_manifold_mix_up_dict_and_label(input_id, train_label)
            else:
                mix_up_dict = None

            output = model(input_id, mask, mix_up_dict)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                if mix_up_type == 'word_mixup':
                    mix_up_dict, val_label = get_word_mix_up_dict_and_label(input_id, val_label)
                elif mix_up_type == 'sentence_mixup':
                    mix_up_dict, val_label = get_sentence_mix_up_dict_and_label(input_id, val_label)
                elif mix_up_type == 'manifold_mixup':
                    mix_up_dict, val_label = get_manifold_mix_up_dict_and_label(input_id, val_label)

                output = model(input_id, mask, mix_up_dict)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
