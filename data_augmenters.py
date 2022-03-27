import numpy as np
import random

MIX_UP_PROB_LENGTH = 3

MIX_UP_TYPES = {
    'sentence': 'sentence_mixup',
    'word': 'word_mixup',
    'manifold': 'manifold_mixup',
}


def get_manifold_mix_up_dict_and_label(input_id, label):
    layers = [i for i in range(MIX_UP_PROB_LENGTH)]
    lam = np.random.beta(2.0, 2.0)
    indices = [i for i in range(input_id.size()[0])]
    random.shuffle(indices)

    mix_up_dict = dict()
    mix_up_dict['layer_num'] = random.choice(layers),
    mix_up_dict['lam'] = lam
    mix_up_dict["shuffled_indices"] = indices

    shuffled_label = label[indices]
    label = (lam * label + (1 - lam) * shuffled_label).long()
    return mix_up_dict, label


def get_word_mix_up_dict_and_label(input_id, label):
    lam = np.random.beta(2.0, 2.0)
    indices = [i for i in range(input_id.size()[0])]
    random.shuffle(indices)

    mix_up_dict = dict()
    mix_up_dict['layer_num'] = 0,
    mix_up_dict['lam'] = lam
    mix_up_dict["shuffled_indices"] = indices

    shuffled_label = label[indices]
    label = (lam * label + (1 - lam) * shuffled_label).long()
    return mix_up_dict, label


def get_sentence_mix_up_dict_and_label(input_id, label):
    lam = np.random.beta(2.0, 2.0)
    indices = [i for i in range(input_id.size()[0])]
    random.shuffle(indices)

    mix_up_dict = dict()
    mix_up_dict['layer_num'] = MIX_UP_PROB_LENGTH - 1,
    mix_up_dict['lam'] = lam
    mix_up_dict["shuffled_indices"] = indices

    shuffled_label = label[indices]
    label = (lam * label + (1 - lam) * shuffled_label).long()
    return mix_up_dict, label


def mix_up(x, current_layer, mix_up_dict, training):
    if training:
        layer_num = mix_up_dict['layer_num']
        if current_layer == layer_num:
            shuffled_indices = mix_up_dict['shuffled_indices']
            lam = mix_up_dict['lam']
            x_shuffled = x[shuffled_indices]
            mixed_x = lam * x + (1 - lam) * x_shuffled
            return mixed_x
        return x
    return x
