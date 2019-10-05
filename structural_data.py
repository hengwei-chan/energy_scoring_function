import os, numpy as np, pickle
import openbabel, pybel, math
import functions as f

def coords(molecule): #retrieving OpenBabel atom types and coordinates of atoms
    atoms = []
    for atom in openbabel.OBMolAtomIter(molecule):
        atype = atom.GetType()
        if atype != 'H':
            atoms.append([atype, atom.GetX(), atom.GetY(), atom.GetZ()])
    return atoms

def atom_pair(a):
    a.sort()
    a = ';'.join(a)
    return a

def data_sdf(path):
    molecule = next(pybel.readfile('sdf', path))
    molecule = molecule.OBMol
    atoms = coords(molecule)
    return atoms

def data_pdb(path):
    text = f.readfile(path, 'l')
    text = ''.join([l for l in text if 'HOH' not in l])
    molecule = pybel.readstring('pdb', text)
    molecule = molecule.OBMol
    atoms = coords(molecule)
    return atoms

def data_extract(receptor_atoms, ligand_atoms):
    close = []
    for rdx, ra in enumerate(receptor_atoms):
        for ldx, la in enumerate(ligand_atoms):
            d = round(f.distance(la[1:], ra[1:]), 2)
            if d <= 5.0:
                close.append([rdx, ldx, d, ra[0], la[0]])
    close = [i[:-2] + [atom_pair(i[-2:])] for i in close]
    return close

def main(directory, cpxs, be):
    data = []
    for cpx in cpxs:
        cpx_en = [i[1:] for i in be if i[0] == cpx]
        if cpx_en == []:
            continue
        lig_atoms = data_sdf(directory + cpx + '/' + cpx + '_ligand.sdf')
        rec_atoms = data_pdb(directory + cpx + '/' + cpx + '_protein.pdb')
        close = data_extract(rec_atoms, lig_atoms)
        if close != []:
            data.append([close, len(lig_atoms), cpx_en[0][1], cpx])
    with open('data_files/initial_structural_data/PDBBind_refined_subset.pkl', 'wb') as ff:
        pickle.dump(data, ff)

tolist = np.ndarray.tolist
prefix = {'fM': 1e-15, 'mM': 1e-3, 'nM': 1e-9, 'pM': 1e-12, 'uM': 1e-6}
directory = 'C:/Users/raulia/Downloads/refined-set/'
#directory_coreset = directory_casf + 'coreset/'
cpxs = list(filter(lambda x: len(x) == 4, os.listdir(directory))) #list of folders containing complexes from database

be = f.readfile(directory + 'index/INDEX_refined_data.2018', 'l')[6:]
be = [i.split() for i in be]
be = [[i[0], float(i[3]), i[4]] for i in be]
be = [i[:-1] + [i[-1].split('=')[1]]  for i in be if '=' in i[-1]]
be = [i[:2] + [round(0.593 * math.log(float(i[-1][:-2]) * prefix[i[-1][-2:]]), 2)] for i in be] #retrieving exp binding energies from database in kcal/mol

main(directory, cpxs, be)
