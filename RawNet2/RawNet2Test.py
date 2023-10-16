import os, librosa, math
import random
from random import shuffle
import numpy as np
from tqdm import tqdm
import pickle as pkl
import soundfile as sf
import torch
import yaml
from rawnetmodel import RawNet, set_random_seed, genSpoof_list, Dataset_gen, produce_evaluation_file
import shutil
from sklearn.metrics import classification_report
import pickle as pkl

def split_samples():
    FoR = "fake"

    in_files = '/data/external/breath/news_articles/' + FoR
    out_files = '/data/external/breath/news_articles_4s/' + FoR

    files = []
    for subdir, dirs, fils in os.walk(in_files):
        for file in fils:
            if (file.__contains__('.wav')):
                files.append(os.path.join(subdir,file))
    mapping = []
    total = 0
    for f in tqdm(files):
        input_audio = f
        sig, rate = librosa.load(input_audio, sr=44100)
        input_audio_name = os.path.basename(input_audio).split(".wav")[0]

        four_sec = 44100 * 4
        num_section = math.ceil(len(sig) / four_sec)
        pad =  (num_section * four_sec) - len(sig)
        pad_sig = np.pad(sig, (0,pad))
        split_sig = np.array(np.split(pad_sig, num_section))
        
        for seg in split_sig:
            total += 1
            sf.write(os.path.join(out_files, str(total))+".wav", seg, 44100)
        
        mapping.append((input_audio, total))

    pkl.dump(mapping,open("saved_objects/news_articles/{}_4s_split_mapping.pkl".format(FoR), "wb"))


def get_total_lengths():
    in_files = '/data/external/breath/news_articles/'
    real = []
    fake = []
    for subdir, _, fils in os.walk(in_files):
        for file in fils:
            if (file.__contains__('.wav')):
                if (subdir.__contains__("/real")):
                    real.append(os.path.join(subdir,file))
                elif (subdir.__contains__("/fake")):
                    fake.append(os.path.join(subdir,file))

    fake_total = 0
    real_total = 0

    for x in real:        
        real_total += librosa.get_duration(filename=x, sr=44100)

    for x in fake:
        fake_total += librosa.get_duration(filename=x, sr=44100)

    ratio = fake_total/real_total
    
    print("\n\nFake duration: {:.0f} min \nReal duration: {:.0f} min \nRatio: {:.3f}".format(fake_total/60, real_total/60, ratio))

def make_protocol():
    files = '/data/external/breath/news_articles_4s/'
    out_files = '/data/external/breath/news_articles_4s_all/'
    fs = []
    for subdir, dirs, fils in os.walk(files):
        FoR = "bonafide" if subdir.__contains__("real") else "spoof"
        for file in fils:
            if (file.__contains__('.wav')):
                fs.append((os.path.basename(os.path.join(out_files,"{}_{}".format(FoR,file))), FoR))
                shutil.copyfile(os.path.join(subdir,file), os.path.join(out_files,"{}_{}".format(FoR,file)))


    protocol_file = "saved_objects/RawNet2/protocol.txt"
    out = open(protocol_file, "w")
    out_str = []
    for f in tqdm(fs):
        writ = "{} - - {}\n".format(f[0].replace(".wav", ""), f[1])
        out_str.append(writ)

    out.writelines(out_str)
    out.close


def eval_rawnet():

    protocol_path = "saved_objects/RawNet2/protocol.txt"
    eval_output = "saved_objects/RawNet2/results/output.txt"
    model_path = "saved_objects/RawNet2/models/model_LA_CCE_100_32_0.0001/epoch_50.pth"
    set_random_seed(1234)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    dir_yaml = os.path.splitext('saved_objects/RawNet2/model_config_RawNet')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.Loader)

    model = RawNet(parser1['model'], device)
    model =(model).to(device)

    model.load_state_dict(torch.load(model_path,map_location=device))
    print('Model loaded : {}'.format(model_path))

    file_eval = genSpoof_list(dir_meta = protocol_path,is_train=False,is_eval=True)
    print('no. of eval trials',len(file_eval))
    eval_set=Dataset_eval(list_IDs = file_eval,base_dir = "/data/external/breath/news_articles_4s_all/")
    produce_evaluation_file(eval_set, model, device, eval_output)


def eval_rawnet_results():

    eval_output = "saved_objects/RawNet2/results/output.txt"
    res = []
    with open(eval_output, 'r') as f:
        res = f.readlines()

    true_y = []
    pred_y = []
    for r in res:
        name,_,pred,_,_ = r.split(' ')
        true = 1
        pr = 1
        if (str(pred)=="1"): #rawnet makes spoof = 0 for some reason
            pr = 0
        if (str(name).__contains__("bonafide")):
            true = 0
        true_y.append(true)
        pred_y.append(pr)

    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    test_res = classification_report(true_y, pred_y, target_names=['real', 'fake'])
    print("\n\n\n-------------------------------------------------------------------")
    print("All news samples")
    print(test_res)


    num_spoof = np.sum(pred_y)
    print("Number predicated as deepfake: {} ".format(num_spoof))
    print("Number predicated as real: {} ".format(pred_y.shape[0]-num_spoof))
    print("-------------------------------------------------------------------")

    real_pred = list(zip(true_y, pred_y))
    shuffle(real_pred)

    temp_true = []
    temp_pred = []
    count = 0
    stop = np.sum(true_y)
    for x in real_pred:
        
        if x[0] == 1:
            temp_true.append(x[0])
            temp_pred.append(x[1])
            
        elif count < stop:
            temp_true.append(x[0])
            temp_pred.append(x[1])
            count+=1

    temp_true = np.array(temp_true)
    temp_pred = np.array(temp_pred)
    temp_test_res = classification_report(temp_true, temp_pred, target_names=['real', 'fake'])
    print("\n\n\n-------------------------------------------------------------------")
    print("Radnomly selected with classes balanced news samples")
    print(temp_test_res)

    temp_num_spoof = np.sum(temp_pred)
    print("Number predicated as deepfake: {} ".format(temp_num_spoof))
    print("Number predicated as real: {} ".format(temp_pred.shape[0]-temp_num_spoof))
    print("-------------------------------------------------------------------")

def split_asvspoof(num_iterations=5):

    fake_sizes = [90,75,50,25,10,1,0.1]
    dir_meta = 'saved_objects/ASVspoof/ASVspoofTrain/ASVspoof2019.LA.cm.train.trn.txt'

    d_real = []
    d_fake = []
    file_list=[]
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        spkr,key,_,atk,label = line.strip().split(' ')
        file_list.append(key)
        
        if label == 'bonafide':
            path = "{}/{}/{}.wav".format(spkr,"bonafide",key)
            d_real.append((key,path))
        else:
            path = "{}/{}/{}.wav".format(spkr,atk,key)
            d_fake.append((key,path))

    num_reals = len(d_real)
    num_fakes =  len(d_fake)
    
    # reals, rpaths = zip(*d_real)
    # d_temp = dict(zip(reals,np.ones(num_reals).astype(np.int16)))

    # fakes, fpaths = zip(*d_fake)
    # fakes = dict(zip(fakes,np.zeros(num_fakes).astype(np.int16)))

    # d_temp.update(fakes)
    # outpath = 'saved_objects/ASVspoof/ASVspoofTrain/allfake_allreal.pkl'
    # file_list = list(d_temp.keys())
    # with open('saved_objects/ASVspoof/ASVspoofTrain/allfake_allreal.lst', "w") as f:
    #     f.writelines("\n".join(file_list))

    # pkl.dump((d_temp,file_list,rpaths + fpaths), open(outpath, 'wb'))
    for i in range(len(fake_sizes)):
        shuffle(d_real)
        d_reals = np.array_split(d_real, num_iterations)
        for iteration, d_rel in enumerate(d_reals):
            num_reals = len(d_rel)
            num_fakes =  np.ceil((num_reals / (1 - fake_sizes[i]/100)) - num_reals).astype(np.int16)
            
            reals, rpaths = zip(*d_rel)
            d_temp = dict(zip(reals,np.ones(num_reals).astype(np.int16)))

            d_fake = np.array(d_fake)[np.random.choice(len(d_fake),num_fakes,replace=False)]
            fakes, fpaths = zip(*d_fake)
            fakes = dict(zip(fakes,np.zeros(num_fakes).astype(np.int16)))

            d_temp.update(fakes)
            outpath = 'saved_objects/ASVspoof/ASVspoofTrain/{}fake_{}real_iter{}.pkl'.format(fake_sizes[i], 100 - fake_sizes[i], iteration)
            file_list = list(d_temp.keys())
            # pkl.dump((d_temp,file_list,rpaths + fpaths), open(outpath, 'wb'))
        


def split_asvspoof_eval(num_iterations=5):

    fake_sizes = [-1,90,75,50,25,10,1]
    dir_meta = 'saved_objects/ASVspoof/ASVspoofEval/trial_metadata.txt'

    d_real = []
    d_fake = []
    file_list=[]
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        _,key,_,_,_,label,_,_ = line.strip().split(' ')
        file_list.append(key)
        
        if label == 'bonafide':
            path = "{}.flac".format(key)
            d_real.append((key,path))
        else:
            path = "{}.flac".format(key)
            d_fake.append((key,path))
            

    num_reals = len(d_real)
    num_fakes = (np.ceil(num_reals / (1 - np.array(fake_sizes)/100)) - num_reals).astype(np.int32)


    for i in range(len(fake_sizes)):
        if fake_sizes[i] == -1:
            reals, rpaths = zip(*d_real)
            d_temp = dict(zip(reals,np.ones(num_reals).astype(np.int16)))

            fakes, fpaths = zip(*d_fake)
            fakes = dict(zip(fakes,np.zeros(len(fakes)).astype(np.int16)))

            d_temp.update(fakes)
            outpath = 'saved_objects/ASVspoof/ASVspoofEval/allfake_allreal.pkl'
            file_list = list(d_temp.keys())
            pkl.dump((d_temp,file_list,rpaths + fpaths), open(outpath, 'wb'))

        else:
            for iteration in range(num_iterations):

                reals, rpaths = zip(*d_real)
                d_temp = dict(zip(reals,np.ones(num_reals).astype(np.int16)))

                d_fake = np.array(d_fake)[np.random.choice(len(d_fake),num_fakes[i],replace=False)]
                fakes, fpaths = zip(*d_fake)
                fakes = dict(zip(fakes,np.zeros(num_fakes[i]).astype(np.int16)))

                d_temp.update(fakes)
                outpath = 'saved_objects/ASVspoof/ASVspoofEval/{}fake_{}real_iter{}.pkl'.format(fake_sizes[i], 100 - fake_sizes[i], iteration)
                file_list = list(d_temp.keys())
                pkl.dump((d_temp,file_list,rpaths + fpaths), open(outpath, 'wb'))





split_asvspoof()

# split_asvspoof_eval()

# make_protocol()
# eval_rawnet()

# eval_rawnet_results()
# get_total_lengths()

