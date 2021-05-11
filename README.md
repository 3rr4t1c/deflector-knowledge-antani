# deflector-knowledge-antani  

Cleaned version of an old messy repo from master's thesis project. Now with more kittens!  

---

## Contents:

This repository contains files and packages for a full run of *Deflector Framework* wich is a system aimed to increase mainly the precision of a relation extraction model. *Deflector* need two pre-trained models: A Relation Extraction model and a Link Prediction model. First, the relation extraction model is used for extracting relations, tracking associations with corresponding pattern (e.g. Whoa! I found 'married to' in between 'Joseph' and 'Maria', I can infer the fact [Joseph, sposue, Maria] cause I know 'married to' mean <spouse> relation, *Deflector record that you know this association and the entities involved 'Joseph' and 'Maria'*). Then the Link Prediction model is exploited in a funky way to reject some of this associations that can be bad.  
Here it is a description of files and folder:

1. `run_test.py` It is the main script. Running it will execute the entire pipeline for testing *Deflector* and produce an output in the `output_data` folder (existing files will be overwrited).
2. `tools.py` It's a module providing some utility for saving, printing and more.
3. `wrappers.py` This module define the wrappers for *Relation Extraction* and *Link Prediction* models.
4. `deflector.py` Define the *Deflector* object used for testing.
5. `LectorPlusPkg` folder. Contains an implementation of *Lector* a model for Relation Extraction wich has been choosen for this project.
6. `ComplExPkg` folder. Contains and implementation of *ComplEx* a model for Link Prediction wich has been choosen for this project. WARNING: the pre-trained model is missing in this repo due to file size limitation of GitHub it was not possible upload such a big file, in future may be provided an external link or a multi-spanned archive to decompress.
7. `input_data` folder. Contains a Knowledge Graph and a loooong list of patterns extracted from wikipedia articles. That's the input of the test pipeline. Lector will use both for training with *Distant Supervision* and ComplEx will use only the Knowledge Graph (KG). The KG is already splitted for ComplEx training, so Lector will use only the `input_data\knowledge_graph\train.txt` part as expected. The original patterns file is very large so a reduced size file with ending "DEMO" name is provided. 
8. `output_data` folder. Will contains the output of test pipeline made of two files. `banned_patterns.tsv` contains a list of all pattern-relation associations that *Deflector* decided to reject using ComplEx. `pattern2relation.tsv` contains a list of all pattern-relation associations that Lector has recognized and used to extracting relations. 
In brief, `banned_patterns.tsv` is a subset of `pattern2relation.tsv`
9. `benchmarks` folder contains a Jupyter Notebook file `evaluation.ipynb` for visualizing the framework performance through some analytics. A script `init_ground_truth.py` will create a `ground_truth.tsv` file in this folder sourcing from `..\output_data\pattern2relation.tsv` with pattern-relation associations sorted by relation names and dummy labels initialized to '2': the user must to manually set this labels to '0' if the association is wrong and should be rejected or '1' otherwise. 

---

## Instruction:

*. *FULL RUN*: Execute `run_test.py`, then `benchmarks\init_ground_truth.py`, open the `ground_truth.tsv` file and manually label all what you want. Run `benchmarks\evaluation.ipynb` to evaluate the framework.
*. *EVALUATION RUN*: If you have already done with manual labelling the ground truth then run `benchmarks\evaluation.ipynb` for visualize the results.

---

``` 
 /\_/\
( o.o )
 > ^ <
 
 
                      /^--^\     /^--^\     /^--^\
                      \____/     \____/     \____/
                     /      \   /      \   /      \
                    |        | |        | |        |
                     \__  __/   \__  __/   \__  __/
|^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
| | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
########################/ /######\ \###########/ /#######################
| | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|

```
