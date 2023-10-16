# RawNet
## Requirements:
## Steps to Reproduce:
1. Install the requirements into an Anaconda Environment.
2. Train the data using the script, *main.py*:
   1. Change the arguments:
      * `--database_path`: the full path to the dataset folder you want to train.
      * `--model_name`: the name you want to give the model, ex: rawnet_train_wavefake.
      * `--protocol_path`: the full path of where the `pkl` file is located.
3.  Score the data using the script, *score_real_world.py*:
   1. Change the arguments:
      * `--database_path`: path of the dataset being scored.
      * `--pretrained`: path of the pretrained model.
      * `--pkl_file`: path of the `pkl` file for the dataset.
      * `--output`: where to output the `.score` file.
