# Wav2Vec

## Requirements:

## Steps to Reproduce:
1. Install the requirements in an Anaconda Environment.
2. Train the data using the script, *main_SSL_DF.py*.
    1. Change the arguments:
         * `protocol_path`: replace with where the produced `pkl` file is located in your data directory.
         * `database_path`: replace with the full path of where your dataset folder is located, ex: /data/FakeAVCeleb or /data/WaveFake.
         * `save_model_dir`: replace with the full path of where you want the model to be saved.
         * `model_name`: replace with a name you would like to give the model, ex: wav2vec_train_fakeavceleb.
    2. Run the script.
3. Score the data using the script, *score_SSL_DF_real-world.py*.
     1. Change the arguments:
        * `--database_path`: the full path location of the dataset folder that you want to score on.
        * `--protocol_path`: the full path of the `pkl` file for the dataset.
        * `--pretrained_path`: the full path of `epoch_99.pth` for the pretrained model from Step 2.
        * `--scores_file`: where to place the output scores file.
     2. Run the script, which should output a `.score` file in your specified directory.
4. Run the `.score` file through the script, *convert_scores_to_probs.py*:
   1. This should produce a `.txt` file in the same directory as the `.score` file.
