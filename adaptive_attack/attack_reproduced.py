import pickle
import random

def merge_nested_dicts(dict1, dict2):
    merged_dict = {}
    
    for key in dict1.keys():
        if key in dict2:
            merged_dict[key] = {**dict1[key], **dict2[key]}
        else:
            merged_dict[key] = dict1[key]
            
    for key in dict2.keys():
        if key not in merged_dict:
            merged_dict[key] = dict2[key]
    
    return merged_dict
with open('./data/adaptive_attack/texture_behavior_attacked.pkl', 'rb') as f:
    attacked_res_1 = pickle.load(f)
with open('./data/adaptive_attack/defenseoutput_attacked.pkl', 'rb') as f:
    attacked_res_2 = pickle.load(f)
with open('./data/adaptive_attack/temporalgraph_attacked.pkl', 'rb') as f:
    attacked_res_3 = pickle.load(f)

with open('./data/adaptive_attack/attack_subset.pkl', 'rb') as f:
    attack_subset = pickle.load(f)
    attack_subset = dict(sorted(attack_subset.items()))

print("Adaptive strategy 1 - Disrupt feature extraction")
pass

filename_attacked_dict = {}
for key, value in attacked_res_1.items():
    for key2, value2 in value.items():
        for box in value2:
            filename_attacked_dict[box[-1]] = True

pygad_attacked = {}
for key, value in attack_subset.items():
    for value2 in value[1]:
        if 'path_adv' in value2 and value2['path_adv'] in filename_attacked_dict and len(value[1]) > 1:
            if key not in pygad_attacked:
                pygad_attacked[key] = {}
                pygad_attacked[key]['len'] = len(value[1])
                pygad_attacked[key]['attacked'] = 0
            pygad_attacked[key]['attacked'] += 1
        

ratios = []
for key, value in pygad_attacked.items():
    r = value['attacked']/value['len']
    if r >= 0.5:
        ratios.append(r)

print(f'{int(len(ratios)/len(attack_subset)*100)}% instance sequences are successfully attacked and evaded our defense consistently.')
print()

print("Adaptive strategy 2 - PhySense output as feedback")
pass

filename_attacked_dict = {}
for key, value in attacked_res_2.items():
    for key2, value2 in value.items():
        for box in value2:
            filename_attacked_dict[box[-1]] = True

pygad_attacked = {}
for key, value in attack_subset.items():
    for value2 in value[1]:
        if 'path_adv' in value2 and value2['path_adv'] in filename_attacked_dict and len(value[1]) > 1:
            if key not in pygad_attacked:
                pygad_attacked[key] = {}
                pygad_attacked[key]['len'] = len(value[1])
                pygad_attacked[key]['attacked'] = 0
            pygad_attacked[key]['attacked'] += 1
        

ratios = []
for key, value in pygad_attacked.items():
    r = value['attacked']/value['len']
    if r >= 0.5:
        ratios.append(r)

print(f'{int(len(ratios)/len(attack_subset)*100)}% instance sequences are successfully attacked and evaded our defense consistently.')
print()
print("Adaptive strategy 3 - Exploit temporal graph")
pass


filename_attacked_dict = {}
for key, value in attacked_res_3.items():
    for key2, value2 in value.items():
        for box in value2:
            filename_attacked_dict[box[-1]] = True

pygad_attacked = {}
for key, value in attack_subset.items():
    for value2 in value[1]:
        if 'path_adv' in value2 and value2['path_adv'] in filename_attacked_dict and len(value[1]) > 1:
            if key not in pygad_attacked:
                pygad_attacked[key] = {}
                pygad_attacked[key]['len'] = len(value[1])
                pygad_attacked[key]['attacked'] = 0
            pygad_attacked[key]['attacked'] += 1
        

ratios = []
for key, value in pygad_attacked.items():
    r = value['attacked']/value['len']
    if r >= 0.5:
        ratios.append(r)

print(f'{int(len(ratios)/len(attack_subset)*100)}% instance sequences are successfully attacked and evaded our defense consistently.')
print()