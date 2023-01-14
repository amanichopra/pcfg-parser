# PCFG Parser

In this project, I use the ATIS subsection of the Penn Treebank corpus to build a PCFG parser. First, I parse the grammer and create a hash table to store the grammar's productions and probabilities. Next, I implement the CKY algorithm to retrieve parse trees. Lastly, I evaluate the parser using F1-score to compare the true parse trees to the predicted parse trees.

Run ```python3 cky.py``` to parse the trees from the training corpus (available [here](https://drive.google.com/file/d/1w5tb_43i5jJs9jJKtTaMaFY9YA0ZYsXJ/view?usp=sharing)).
