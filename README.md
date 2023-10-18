# Better Be Computer
## Overview
## Requirements:
## Data:
We provide the datasets required to reproduce our deepfake detectors.  Our full, collected dataset for human response themes is also included, named *responses.csv*, located in the *fmo_responses* folder.
## Steps to Reproduce:
1. Download the repository, such as: 
```
git clone https://github.com/fakemeout-paper/fake-me-out.git
```
2. Install the requirements:
```
pip install -r requirements.txt
```
3. Follow the *README* instructions within each model folder.
4.  Run the `.score` file for each model through the Python script, *convert_scores_to_probs.py*:
     *  Change `scores = ''` to the full path of where the `.score` produced from Step 3.
     *  This should output a similarily named file in the same directory as the `.score` file, but with the `.txt` extension.

## Citation:
