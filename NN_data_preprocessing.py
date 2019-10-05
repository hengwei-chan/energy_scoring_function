import pandas as pd, numpy as np, pickle, os

def filter_cpx(data, k):
    data_red = []
    for idx in range(len(data)):
        unique = {}
        for adx, a in enumerate(data[idx][0]):
            if a[-1] not in unique:
                unique[a[-1]] = [a[0]]
            else:
                unique[a[-1]].append(a[0])
        max_unique = max([len(unique[i]) for i in unique])
        if max_unique <=k:
            data_red.append(data[idx])
    return data_red

def tables(name, mol_atoms, atypes, k):
    ranges = list(range(0, len(mol_atoms), 500))
    ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], len(mol_atoms)]]
    for m in range(k):
        print(m)
        load_files = []
        for rdx, r in enumerate(ranges):
            load_file = name + str(m) + '_' + str(rdx) + '.txt'
            load_files.append(load_file)
            df = pd.DataFrame(columns=atypes)
            for i in mol_atoms[r[0]:r[1]]:
                unique = {}
                for adx, a in enumerate(i):
                    if a[-1] not in unique:
                        unique[a[-1]] = [a[2]]
                    else:
                        unique[a[-1]].append(a[2])

                row_input = {}
                for a in atypes:
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

def main(data, name):
    k = 130
    data_red = filter_cpx(data, k) #choising of complexes having no more than 130 atom pairs of the same type
    atom_pairs = [i[0] for i in data_red]
    num_ligand_atoms = np.array([i[1] for i in data_red])
    exp_binding_energy = np.array([i[2] for i in data_red])
    complex_name = np.array([i[3] for i in data_red])
    atom_pair_types = np.unique(np.array([a for m in [[j[-1] for j in i] for i in atom_pairs] for a in m]))
    tables('data_files/NNdata/' + name, atom_pairs, atom_pair_types, k)
    tables2('data_files/NNdata/' + name + 'd', [num_ligand_atoms, exp_binding_energy, complex_name])

with open('data_files/initial_structural_data/PDBBind_refined_subset.pkl', 'rb') as ff:
    train_data = pickle.load(ff)
with open('data_files/initial_structural_data/casf2016.pkl', 'rb') as ff:
    test_data = pickle.load(ff)

main(train_data, 'train')
main(test_data, 'test')
