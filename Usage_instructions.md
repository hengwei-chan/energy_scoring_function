<strong>1.	Retrieving of the descriptors from protein-ligand complexes.</strong>

Main descriptor is a list of atom pairs defined as two atoms within 5 Å and the corresponding distances between these atoms. Atom types should be were assigned by the Open Babel software[1], an additional atom type ‘HN’ was introduced for denoting of hydrogens connected to nitrogen atoms of amino groups. The atom pair labels were defined by concatenating the atomic types of the contributing atoms sorted alphabetically with the semicolon symbol used as a separator, e.g. ‘C+;C3’. Second descriptor is a total number of  heavy atoms and hydrogens connected to oxygen atoms or nitrogen atoms of amino groups of the ligand.

Example of input: [[['C3;Car', 4.4711], ['C3;Car', 4.6541], ['C3;O3', 4.2251], ['C3;HO', 4.8206]], 14], where 14 - number of ligand atoms.

This step can be conducted using structural_data.py script. The command is “python structural_data.py directory name”, where directory is a folder containing structural files. It is supposed that the files are organized in folders containing separate files for theligand and the protein binding pocket. Recommended format for ligand - .sdf. "Name" is a name of output file without extention.

<strong>2.	Prediction of binding energy.</strong>

SF_hard _cutoffs.py and SF_soft _cutoffs.py files contain scripts for calculations. Command: “python SF_soft _cutoffs.py file k”, where file is the file containing descriptors and k – number of distance interval for counting of atom pairs. 


[1] Boyle, N. M. O., Banck, M., James, C. A., Morley, C., Vandermeersch, T., & Hutchison, G. R. (2011). Open Babel: An open chemical toolbox. Journal of Cheminformatics, 3(33), 1–14. https://doi.org/10.1186/1758-2946-3-33
