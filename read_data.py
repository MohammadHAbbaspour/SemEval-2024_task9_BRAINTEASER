import numpy as np
import sys

base_dir = sys.argv[1] if len(sys.argv) > 1 else 'F:\\NLP-Lab\\Internship\\SemEval\\SemEval-2024_task9_BRAINTEASER\\'

def read_data(path, allow_pickle=True):
    data = np.load(path, allow_pickle=allow_pickle)
    return data

def count_unsure_choices(data):
    count = 0
    for i in range(len(data)):
        if 'None of above.' in data[i]['choice_list']:
            count += 1
    return count

sp_train_data = read_data(base_dir + 'SP-train.npy')
print(count_unsure_choices(sp_train_data) == sp_train_data.shape[0])

wp_train_data = read_data(base_dir + 'WP-train.npy')
print(count_unsure_choices(wp_train_data) == wp_train_data.shape[0])

sp_rnd_idx = np.random.randint(0, sp_train_data.shape[0])
wp_rnd_idx = np.random.randint(0, wp_train_data.shape[0])

print(sp_train_data[sp_rnd_idx], '\n')
print(wp_train_data[wp_rnd_idx])