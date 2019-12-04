import os, numpy as np, pickle, sys
import openbabel, pybel

def distance(i, j):
    d = ((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2 + (i[2] - j[2]) ** 2) ** 0.5
    return d

def coords(molecule): #retrieving OpenBabel atom types and coordinates of atoms
    atoms = []
    for atom in openbabel.OBMolAtomIter(molecule):
        atype = atom.GetType()
        if atype != 'H':
            atoms.append([atype, atom.GetX(), atom.GetY(), atom.GetZ()])
        else:
            for neigh in openbabel.OBAtomAtomIter(atom):
                ntype = neigh.GetType()
                if ntype == 'Nam':
                    atoms.append(['HNam', atom.GetX(), atom.GetY(), atom.GetZ()])
    return atoms

def atom_pair(a):
    a.sort()
    a = ';'.join(a)
    return a

def data_sdf(path):
    molecule = next(obenbabel.readfile('sdf', path))
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
            if not ((la[0][0] == 'H') and (ra[0][0] == 'H')):
                d = round(distance(la[1:], ra[1:]), 4)
                if d <= 5.0:
                    close.append([d, ra[0], la[0]])
    close = [[atom_pair(i[-2:]), i[0]] for i in close]
    return close

def main(directory, name):
    data = []
    cpxs = os.listdir(directory)
    #Structural data should be organized in folders containing separate files for ligand and protein binding pocket.
    #Recommended format for ligand - .sdf
    for idx, cpx in enumerate(cpxs):
        lig_atoms = data_sdf(directory + cpx[0] + '/' + cpx[0] + '_ligand.sdf')
        rec_atoms = data_pdb(directory + cpx[0] + '/' + cpx[0] + '_protein.pdb')
        close = data_extract(rec_atoms, lig_atoms)
        if close != []:
            data.append([close, len(lig_atoms)])
    with open(name + '.pkl', 'wb') as ff:
        pickle.dump(data, ff)

directory = sys.argv[1]
name = sys.argv[2]
main(directory, name)

