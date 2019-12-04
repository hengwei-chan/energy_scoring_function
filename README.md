# CBSF
Scoring function for estimation of binding energies between proteins and small molecules.

A new CBSF empirical scoring function for the estimation of binding energies between proteins and small molecules is presented in this project. Scoring function is based on counting the protein-ligand interacting atom pairs, the counting can be conducted both in a single interval and a few (2-4) intervals. Additional descriptor includes a number of heavy atoms and hydrogens connected to oxygen atoms or nitrogen atoms of amino groups of the ligand.
There are two versions of the scoring functions: with soft cutoffs for counting of the number of atom pairs and with hard cutoffs. In hard cutoff versions number of atom pairs are calculated according formula:
![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/f1.png)

where A and B are the atoms at the ligand (L)-protein (P) interface, R<sub>AB</sub> is the interatomic distance between atoms A and B, R<sup>0</sup> (T<sub>A</sub>;T<sub>B</sub>) is the distance cutoff for the atom types of A (i.e. T<sub>A</sub>) and B (i.e. T<sub>A</sub>).
![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/f3.png)
![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/soft_cutoffs.png)
![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/hard_cutoffs.png)
