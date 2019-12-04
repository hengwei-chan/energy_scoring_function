import pandas as pd, numpy as np, pickle, os

def filter_cpx(data, k):
    data_red = []
    for idx in range(len(data)):
        unique = {}
        for adx, a in enumerate(data[idx][0]):
            if a[0] not in unique:
                unique[a[0]] = [a[1]]
            else:
                unique[a[0]].append(a[1])
        max_unique = max([len(unique[i]) for i in unique])
        if max_unique <=k:
            data_red.append(data[idx])
    return data_red

def tables(name, atom_pairs, atom_pair_types, k):
    ranges = list(range(0, len(atom_pairs), 500))
    ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], len(atom_pairs)]]
    for m in range(k):
        #lens = {}
        print(m)
        load_files = []
        for rdx, r in enumerate(ranges):
            load_file = name + str(m) + '_' + str(rdx) + '.txt'
            load_files.append(load_file)
            df = pd.DataFrame(columns=atom_pair_types)
            for i in atom_pairs[r[0]:r[1]]:
                unique = {}
                for adx, a in enumerate(i):
                    if a[0] not in unique:
                        unique[a[0]] = [a[1]]
                    else:
                        unique[a[0]].append(a[1])
                row_input = {}
                for a in atom_pair_types:
                    if (a in unique) and (len(unique[a]) > m):
                        row_input[a] = unique[a][m]
                    else:
                        row_input[a] = 100.0
                df = df.append(row_input, ignore_index=True)
            print(m, rdx)
            df.to_csv(load_file, sep=' ', mode='w')
        frames = [pd.read_csv(load_file, sep=' ') for load_file in load_files]
        result = pd.concat(frames)
        result = result.drop(['Unnamed: 0'], axis=1)
        result.to_csv(name + str(m) + '.csv', sep=' ', mode='w')
        for load_file in load_files:
            os.remove(load_file)

def tables2(name, cp):
    df = pd.DataFrame(columns=['num_ligand_atoms', 'exp_binding_energy', 'complex_name'])
    df['num_ligand_atoms'], df['exp_binding_energy'], df['complex_name'] = cp[0], cp[1], cp[2]
    df.to_csv(name + '.csv', sep=' ', mode='w')

def main(data, name, atom_pair_types):
    k1, k2, k3 = 130, 10, 19
    data_red = filter_cpx(data, k1)
    print(len(data), len(data_red))
    atom_pairs = [i[0] for i in data_red]
    num_ligand_atoms = np.array([i[1] for i in data_red])
    exp_binding_energy = np.array([i[2] for i in data_red])
    complex_name = np.array([i[3] for i in data_red])
    if atom_pair_types == []:
        atom_pair_types, counts = np.unique(np.array([a for m in [[j[0] for j in i] for i in atom_pairs] for a in m]), return_counts=True)
        freq = {}
        for apt in atom_pair_types:
            freq_i = np.array([len([j[0] for j in i if j[0] == apt]) for i in atom_pairs])
            freq_i = max(freq_i)
            freq[apt] = freq_i

        atom_pair_types1 = [i for i in freq if freq[i] > 10]
        atom_pair_types2 = [i for i in freq if i not in atom_pair_types1]
        atom_pair_types = [atom_pair_types1, atom_pair_types2]
        print(len(atom_pair_types1), len(atom_pair_types2))

    else:
        tables('data_files/NNdata/' + name + '_1_', atom_pairs, atom_pair_types[0], k1)
        tables('data_files/NNdata/' + name + '_2_', atom_pairs, atom_pair_types[1], k2)
        tables2('data_files/NNdata/' + name + 'd', [num_ligand_atoms, exp_binding_energy, complex_name])
    return atom_pair_types

with open('data_files/initial_structural_data/PDBBind_refined_subset_full.pkl', 'rb') as ff:
    train_data_ref = pickle.load(ff)
with open('data_files/initial_structural_data/casf2016_full.pkl', 'rb') as ff:
    test_data = pickle.load(ff)

train_data_ref.sort(key=lambda x: x[-2])
train_data_ref = [i for i in train_data_ref if i[-2] not in [j[-2] for j in test_data]]

atom_pair_types = main(train_data_ref + test_data, '', [])
#print(len(atom_pair_types[0]), len(atom_pair_types[1]), len(atom_pair_types[2])) 
atom_pair_types = main(train_data_ref, 'ref_train', atom_pair_types)
atom_pair_types = main(test_data, 'test', atom_pair_types)

