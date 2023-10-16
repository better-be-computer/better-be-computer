# CQCC-GMM 

## Requirements:

## Steps to Reproduce:
1. Install the requirements in an Anaconda Environment.
2. Train the data using the script, *asvspoof2021_baseline.py*:
   1. Change the arguments:
      * `--database_path`: the folder containing the dataset you want to train.
      * `--name`: what you want to name the model.
      * `--protocol_path`: the path containing the `pkl` file.
3. Run the file, *scoring_pretrain.py*, to score the data.
   1. This should output a `.score` file into your specified directory.
4. Run the `.score` file through the Python script, *convert_scores_to_probs.py*:
   1. Change `scores = ''` to the full path of where the `.score` file from Step 3 is located.
   2. This should produce a similarily named file in the same directory as the `.score` file, but with the `.txt` extension.
