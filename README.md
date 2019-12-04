# CBSF
<h2>Scoring function for estimation of binding energies between proteins and small molecules.</h2>

A new CBSF empirical scoring function for the estimation of binding energies between proteins and small molecules is presented in this project. Scoring function is based on counting the protein-ligand interacting atom pairs, the counting can be conducted both in a single interval and a few (2-4) intervals. Additional descriptor includes a number of heavy atoms and hydrogens connected to oxygen atoms or nitrogen atoms of amino groups of the ligand.
There are two versions of the scoring functions: with soft cutoffs for counting of the number of atom pairs and with hard cutoffs. In hard cutoff versions number of atom pairs are calculated according formula:

![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/f1.png)

where A and B are the atoms at the ligand (L)-protein (P) interface, R<sub>AB</sub> is the interatomic distance between atoms A and B, R<sup>0</sup> (T<sub>A</sub>;T<sub>B</sub>) is the distance cutoff for the atom types of A (i.e. T<sub>A</sub>) and B (i.e. T<sub>A</sub>).
In soft cutoff versions number of atom pairs are calculated according by taking into account a transition zone of 0.5 Å if the values of R<sup>0</sup> (T<sub>A</sub>;T<sub>B</sub>) and R<sub>AB</sub> are close:
![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/f3.png)

Scoring functions were tested on 285 complexes of CASF-2016 benchmark set [1]. These complexes were excluded from the training set during development of the method. Performance of developed scoring functions on the test set complexes comprising mean absolute error (MAE), median error, Pearson’s correlation coefficient R, standard deviation (\sigma), Spearman correlation coefficient (SP), Kendall correlation coefficient (τ) and predictive index (PI) are presented in Figure 1-2.
![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/soft_cutoffs.png)

Figure 1. Accuracy of the scoring functions with soft cutoffs tested on CASF-2016 benchmark
![formula1](https://github.com/rsyrlyb/CBSF/blob/master/Figures/hard_cutoffs.png)

Figure 2. Accuracy of the scoring functions with hard cutoffs tested on CASF-2016 benchmark

Distinguishing feature of the presented empirical scoring functions is that all parameters and coefficients of the scoring functions are found by means of the neural networks.  The architecture of the neural networks can be found in “files_Neural_network”.

[1] Su, M., Yang, Q., Du, Y., Feng, G., Liu, Z., Li, Y., & Wang, R. (2019). Comparative Assessment of Scoring Functions: The CASF-2016 Update. Journal of Chemical Information and Modeling, 59(2), 895–913. https://doi.org/10.1021/acs.jcim.8b00545
