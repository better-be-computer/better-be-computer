import argparse
from tqdm import tqdm
from gmm import train_gmm
from os.path import exists
import pickle, os
import pickle as pkl
import numpy as np
import h5py


def load_split_dict_og(pth):

    
    d_meta, file_list, path = pkl.load(open(pth, 'rb'))

    return d_meta, file_list, path

def load_split_dict(number):

    # change audio location for datasets
    audio_location = "/data/WaveFake/pkl/pkl.pkl"
    process_files = []
    for _, _, files in os.walk(audio_location):
        for file in files:
            if (".pkl" in file):
                process_files.append(os.path.join(audio_location,file))

    process_files = sorted(process_files)
    number = int(number)
    print(process_files[number])
    d_meta, file_list, path = pkl.load(open(process_files[number], 'rb'))

    return d_meta, file_list, path

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def combine_features():
    
    location = "/CQCC-GMM/old_features/features"
    process_files = []
    for subdir, _, files in os.walk(location):
        for file in files:
            if (".h5" in file):
                process_files.append(os.path.join(subdir,file))

    process_files = sorted(process_files)

    
    with h5py.File("/CQCC-GMM/features/training_cqcc-gmm.h5", "w") as f:
        for file in tqdm(process_files):
            with h5py.File(file, "r") as f1:
                Datasetnames=get_dataset_keys(f1)
                for x in tqdm(Datasetnames):
                    name = os.path.basename(x).replace(".wav","")
                    group = f1.get(x)
                    if group is not None:
                        data = group[()]
                        f.get(name) or f.create_dataset(name, data=data, compression='gzip')

    return process_files


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--protocol_path', type=str, default='')
    
  
    # parser.add_argument('--iteration_numbers', type=str, default="0-5")
    args = parser.parse_args()

    # configs - feature extraction e.g., LFCC or CQCC
    features = 'cqcc'

    # configs - GMM parameters
    ncomp = 512

    

    # configs - train & dev data - if you change these datasets
    db_folder = args.database_path
    
    audio_ext = '.wav'
    # iterations_start, iterations_stop = args.iteration_numbers.split("-")
    # iterations = np.arange(int(iterations_start), int(iterations_stop))
    # for it_num in tqdm(iterations):
    d_label_trn,_,path = load_split_dict_og(args.protocol_path)

    key_list = np.array(list(d_label_trn.keys()))
    val_list = np.array(list(d_label_trn.values()))

    rpositions = np.where(val_list == 1)[0]
    rkeys = key_list[rpositions]
    rpaths = np.array(path)[rpositions]

    fpositions = np.where(val_list == 0)[0]
    fkeys = key_list[fpositions]
    fpaths = np.array(path)[fpositions]


    # GMM pkl file
    dict_file_bona = "/CQCC-GMM/features/gmm_LA_cqcc.pkl"
    dict_file_final = "/CQCC-GMM/final/gmm_cqcc_{}.pkl".format(args.name)
    dict_file = "/CQCC-GMM/features/gmm_LA_cqcc_{}.pkl".format(args.name)
    
    # train bona fide & spoof GMMs
    if not exists(dict_file):
        gmm_bona = train_gmm(data_label='bonafide', features=features,
                            train_keys=rkeys, train_folders=rpaths, audio_ext=audio_ext,
                            dict_file=dict_file, ncomp=ncomp,
                            init_only=True, db_folder=db_folder,name="-bona-{}".format(args.name))
        gmm_spoof = train_gmm(data_label='spoof', features=features,
                            train_keys=fkeys, train_folders=fpaths, audio_ext=audio_ext,
                            dict_file=dict_file, ncomp=ncomp,
                            init_only=True, db_folder=db_folder,name="-spoof-{}".format(args.name))

        gmm_dict = dict()
        gmm_dict['bona'] = gmm_bona._get_parameters()
        gmm_dict['spoof'] = gmm_spoof._get_parameters()
        with open(dict_file, "wb") as tf:
            pkl.dump(gmm_dict, tf)


    gmm_dict = dict()
    with open(dict_file + '_bonafide_init_partial.pkl', "rb") as tf:
        gmm_dict['bona'] = pkl.load(tf)

    with open(dict_file + '_spoof_init_partial.pkl', "rb") as tf:
        gmm_dict['spoof'] = pkl.load(tf)

    with open(dict_file_final, "wb") as f:
        pkl.dump(gmm_dict, f)

