import argparse
from math import ceil, floor
from threading import Thread

from tqdm import tqdm
from gmm import scoring, extract_cqcc
import pickle as pkl
import os
import h5py
import numpy as np

def load_model(number):

    audio_location = "/CQCC-GMM/final"
    process_files = []
    for _, _, files in os.walk(audio_location):
        for file in files:
            if (".pkl" in file):
                process_files.append(os.path.join(audio_location,file))

    process_files = sorted(process_files)
    number = int(number)
    print(process_files[number])
    return process_files[number]

def load_eval_pkl():

    audio_location = "/saved_objects/ASVspoof/ASVspoofEval"
    process_files = []
    for _, _, files in os.walk(audio_location):
        for file in files:
            if ("allreal.pkl" in file):
                process_files.append(os.path.join(audio_location,file))

    process_files = sorted(process_files, reverse=True)

    return process_files


def extract_feats(num):
    def extract(it):  
        evals = eval_files[it]
        for i, file in enumerate(tqdm(evals)):
            
            # cqcc is very slow, writing entire dataset to hdf5 file beforehand (offline cache)
            cache_file = os.path.join("/CQCC-GMM/features_eval",'{}.h5'.format(it))
            with h5py.File(cache_file, 'a') as h5:
                group = h5.get(file)
                if group is None:
                    data = extract_cqcc(file).astype(np.float32)
                    h5.create_dataset(file, data=data, compression='gzip')
                else:
                    data = group[()]


    path = ''
    files = load_eval_pkl()[0]
    _,_,eval_files = pkl.load(open(files, "rb"))
    eval_files = np.array(eval_files)
    eval_files = np.core.defchararray.add(path,eval_files)
    ind = np.arange(0,len(eval_files), ceil((len(eval_files))/25))[1:]
    eval_files = np.split(eval_files, ind)

    extract(num)


def merge_h5():
    loc = "/CQCC-GMM/features_eval"
    process_files = []
    for _, _, files in os.walk(loc):
        for file in files:
            if (".h5" in file):
                with h5py.File(os.path.join(loc,file), "r") as f:
                    # Print all root level object names (aka keys) 
                    # these can be group or dataset names 
                    # print("Keys: %s" % f.keys())
                    # # get first object name/key; may or may NOT be a group
                    a_group_key = list(f.keys())[0]

                    # # get the object type for a_group_key: usually group or dataset
                    # print(type(f[a_group_key])) 

                    # # If a_group_key is a group name, 
                    # # this gets the object names in the group and returns as a list
                    # data = list(f[a_group_key])

                    # # If a_group_key is a dataset name, 
                    # # this gets the dataset values and returns as a list
                    # data = list(f[a_group_key])
                    # # preferred methods to get dataset values:
                    # ds_obj = f[a_group_key]      # returns as a h5py dataset object
                    ds_arr = f[a_group_key][()]  # returns as a numpy array
                    process_files.extend(ds_arr)

    process_files = np.array(process_files)
    with h5py.File(os.path.join(loc,"cqccgmm.h5"), "w") as data_file:
        data_file.create_dataset("dataset_name", data=process_files)

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def check_features():
    loc = "/CQCC-GMM/features_eval"
    process_files = []
    for _, _, files in os.walk(loc):
        for file in files:
            if (".h5" in file):
                process_files.append(file)
    # with h5py.File(final_h5, 'a') as out:           
    for file in tqdm.tqdm(process_files):
        with h5py.File(os.path.join(loc,file), "r") as f:
                    
            Datasetnames=get_dataset_keys(f)
            for name in Datasetnames:
                base = os.path.basename(name).replace(".flac","")
                group = f.get(name)
                if group is not None:
                    data = group[()]
                    # out.create_dataset(base, data=data, compression='gzip')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    parser.add_argument('--x', type=int, default=0)
    args = parser.parse_args()
    extract_feats(args.x)
    # Dataset path
    parser.add_argument('--database_path', type=str, default='/data/FakeAVCeleb')

    parser.add_argument('--iteration_numbers', type=str, default="0-5")
    args = parser.parse_args()

    iterations_start, iterations_stop = args.iteration_numbers.split("-")
    iterations = np.arange(int(iterations_start), int(iterations_stop))
    train_map = ["75fake25real", "50fake50real", "25fake75real", "10fake90real", "1fake99real", "90fake10real"]
    train_map = sorted(train_map)
    eval_file = load_eval_pkl()
    eval_path = args.database_path
    train_split = train_map[int(iterations_start)]
    dicts = []
    for it_num in iterations:        
        dict_file = load_model(it_num)
        dicts.append(dict_file)

    features = 'cqcc'
    db_folder = args.database_path  # put your database root path here
    eval_folder = db_folder   
    audio_ext = '.wav'
           
    eval_name = "allreal_allfake"
    scores_file = "/CQCC-GMM/results/train-{}_iter{}___eval-{}.scores"
    # run on ASVspoof 2021 evaluation set
    scoring(scores_file=scores_file, dict_files=dicts, files=eval_file[0], path=eval_path, features=features,features_cached=True, name=train_split)
