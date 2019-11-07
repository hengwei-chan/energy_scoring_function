import pandas as pd, numpy as np, pickle, os

atom_types = ['C-', 'O-', 'C+', 'N3+','Ng+', 'C1', 'N1', 'C2', 'Car', 'N2', 'Nar', 'S2', 'O2', 'N3','Nam', 'O3', 'O3H',
              'Npl', 'S3', 'Cl', 'F', 'Br', 'I', 'Sox', 'So2', 'Nox', 'Ntr', 'Pac', 'P', 'Sac', 'Cac', 'HO', 'O.co2',
              'C3','Co', 'Cu', 'Mn', 'Ni', 'Zn', 'Fe', 'Mg', 'Na', 'K', 'Li', 'Ca', 'Sr', 'Cs', 'Cd', 'Hg', 'Se']

def filter_cpx(data):
    data_red = []
    for idx in range(len(data)):
        unique = {}
        for adx, a in enumerate(data[idx][0]):
            if a[-1] not in unique:
                unique[a[-1]] = [a[0]]
            else:
                unique[a[-1]].append(a[0])
        max_unique = max([len(unique[i]) for i in unique])
        if max_unique <=130:
            data_red.append(data[idx])
    return data_red

def tables(name, mol_atoms, atypes):
    ranges = list(range(0, len(mol_atoms), 500))
    ranges = [[ranges[i], ranges[i + 1]] for i in range(0, len(ranges) - 1)] + [[ranges[-1], len(mol_atoms)]]
    for m in range(130):
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

def reduce(bonds):
    red_bonds = []
    for idx, bond in enumerate(bonds):
        i = [a.split(',') for a in bond[-1].split(';')]
        b = [a[0] for a in i] # + ',' + str(len(a[1:])) if len(a[1:])>1 else a[0]
        b.sort()
        m = ';'.join(b)
        bond = bond[:-1] + [m]
        red_bonds.append(bond)
    return red_bonds

def split_train_text(data, k):
    close_atoms_amn = [[j[-1] for j in i[0]] for i in data]
    atom_types_amn, count = np.unique(np.array([a for m in close_atoms_amn for a in m]), return_counts=True)
    atom_types_amn_dict = {}
    for idx, i in enumerate(atom_types_amn):
        atom_types_amn_dict[i] = count[idx]
    n = 0
    train, test = [], []
    for idx, i in enumerate(data):
        types = [j[-1] for j in i[0]]
        types_red = [j for j in types if atom_types_amn_dict[j] > k]
        if (len(types) == len(types_red)) and (len(test) < 500):
            test.append(data[idx])
            n += 1
        else:
            train.append(data[idx])
    print(len(train), len(test))
    return train, test, atom_types_amn

def make_tables_d2(name, cp):
    df = pd.DataFrame(columns=['la', 'mw', 'rot', 'hb', 'solv', 'ki', 'cpx'])
    df['la'], df['mw'], df['rot'], df['hb'], df['solv'], df['ki'], df['cpx'] = cp[0], cp[3], cp[4], cp[5], cp[6], cp[1], cp[2]
    df.to_csv(name + '.csv', sep=' ', mode='w')

def main(data, name):
    data_red = filter_cpx(data)

    conn = [i[0] for i in data_red]
    la = np.array([i[1] for i in data_red])
    ki = np.array([i[2] for i in data_red])
    cpx = np.array([i[3] for i in data_red])
    mw = np.array([i[4] for i in data_red])
    rot = np.array([i[5] for i in data_red])
    hb = np.array([i[6] for i in data_red])
    solv = np.array([i[7] for i in data_red])

    atypes = np.unique(np.array([a for m in [[j[-1] for j in i] for i in conn] for a in m]))
    print(len(data), len(data_red), len(atypes))

    tables('txt/vwr_130/' + name, conn, atypes)
    make_tables_d2('txt/vwr_130/' + name + 'd', [la, ki, cpx, mw, rot, hb, solv])

with open('txt/raw_data/refined.pkl', 'rb') as ff:
    data1 = pickle.load(ff)
with open('txt/raw_data/casf2016.pkl', 'rb') as ff:
    data2 = pickle.load(ff)
data1 = [[reduce(i[0])] + i[1:] for i in data1]
data2 = [[reduce(i[0])] + i[1:] for i in data2]

main(data1, 'train')
main(data2, 'test')
