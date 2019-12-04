import pickle, sys

structures = sys.argv[1]
#Structures is a file containig list of complexes represented by their atom pairs and the number of atoms in the ligand, for example:
#[[['C3;Car', 4.4711], ['C3;Car', 4.6541], ['C3;O3', 4.2251], ['C3;HO', 4.8206]], 14], where 14 - number of ligand atoms
#The protein-ligand atom pair was is defined as two atoms within 5 Å
#The atom pairs can be retrieved from pdb files using structural_data.py

ki = sys.argv[2] # ki - number of intervals of distances taking into account by the scoring function; k=1-4
with open(structures, 'rb') as ff:
    complexes = pickle.load(ff)
with open('parameters_SF/SF' + str(ki) +'_parameters.pkl', 'rb') as fl:
    w = pickle.load(fl)

for complex in complexes:
    int_ergies = []
    pairs = complex[0]
    p_ints = []
    for dx, dict in enumerate(w):
        p_int = {}
        for pair in pairs:
            c = 0
            k = (dict[pair[0]][0]-pair[1])/5
            if k > 0.05:
                c = 1
            elif (k <=0.05) and (k>=-0.05):
                c = 10*k+0.5
            if c > 0:
                if pair[0] not in p_int:
                    p_int[pair[0]] = c
                else:
                    p_int[pair[0]] += c
        if dx != 0:
            for pair in p_int:
                cu = sum([i[pair] for i in p_ints if pair in i])
                p_int[pair] -= cu
        p_ints.append(p_int)
        g1 = sum([p_int[i] for i in p_int]) * dict['n1'][0] + dict['n1'][1]
        g2 = sum([p_int[i] for i in p_int]) * dict['n2'] / complex[1]
        g3 = sum([p_int[i]*dict[i][1] for i in p_int])
        g_int = g1 + g2 + g3
        int_ergies.append(g_int)
    g = round(sum(int_ergies), 2) #g is predicted binding free energy

