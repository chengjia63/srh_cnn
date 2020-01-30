import json
from collections import OrderedDict
import numpy as np


def get_patient_list(data):
    patient_set = set()
    uniq_patients = []
    for p in [k.split('/')[1] for k,_ in data.items()]:
        if p not in patient_set:
            uniq_patients.append(p)
            patient_set.add(p)
    return uniq_patients

def get_patient_slices(data, patients):
    out = dict([(p, []) for p in patients])
    for k, _ in data.items():
        out[k.split('/')[1]].append(k)
    return out

def get_first_instance(data, patients):
    out = dict()
    for p in patients.keys():
        min_ = '99999999.tif'
        file_count = 0
        for i in patients[p]:
            if isinstance(data[i], list):
                one_slice_list = [d['fname'] for d in data[i] if len(d['fname'])]
                min_ = min([min_] + one_slice_list)
                file_count += len(one_slice_list)
            else:
                if (len(data[i]['fname'])):
                    min_ = min([min_, data[i]['fname']])
                    file_count += 1
        if file_count:
            out[p] = {'fmin': min_, 'fcount': file_count}
    return out

def main():
    with open('carcinoma_file_mapping.json') as fd:
    #with open('lymphoma_file_mapping.json') as fd:
        data = json.load(fd, object_pairs_hook=OrderedDict)

    uniq_patients = get_patient_list(data)
    pat_slices = get_patient_slices(data, uniq_patients)
    patient_data = get_first_instance(data, pat_slices)
    patient_data = sorted(patient_data.items(), key=lambda x: x[1]['fmin'])
    
    num_grids = np.array([p[1]['fcount'] for p in patient_data])
    all_grids = sum(num_grids)
    cum_reverse = np.cumsum(np.flip(num_grids))

    # the separation are just a suggestion
    valid_start = len(patient_data) - np.argwhere(cum_reverse > .30 * all_grids)[0].item()
    test_start = len(patient_data) - np.argwhere(cum_reverse > .15 * all_grids)[0].item()
    
    fold = ['train'] * valid_start + ['valid'] * (test_start-valid_start) + \
        ['test'] * (len(patient_data) - test_start) 

    print('patient, begin_patch, num_patch, fold')
    for i in range(len(patient_data)):
        print('{}, {}, {}, {}'.format(patient_data[i][0], 
            patient_data[i][1]['fmin'], patient_data[i][1]['fcount'], fold[i]))

    print('@@@ perct train: {:1.3f}'.format(
        sum(num_grids[[f=='train' for f in fold]]) / all_grids))
    print('@@@ perct valid: {:1.3f}'.format(
        sum(num_grids[[f=='valid' for f in fold]]) / all_grids))
    print('@@@ perct test: {:1.3f}'.format(
        sum(num_grids[[f=='test' for f in fold]]) / all_grids))


if __name__ == '__main__':
    main()