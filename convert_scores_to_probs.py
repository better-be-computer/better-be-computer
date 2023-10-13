import numpy as np
import os
import torch
from sklearn.metrics import classification_report
import pickle as pkl

def probs_convert(scores_in):
    replace = f".{scores_in.split('.')[-1]}"
    scores_out = scores_in.replace(replace, ".txt")
    lines = []
    with open(scores_in, "r") as f:
        lines = f.readlines()
    # lines = pkl.load(open(scores_in, 'rb'))

    if "RawNet2" in scores_in:
        with open(scores_out, "w") as fh:
            fh.writelines(lines)
        return

    scr = []
    for x in lines:
        if "lfcclcnn" in scores_in:
            _, name, _,  score = x.split(", ")
            score = score.replace("\n", "")
        else:
            name, score = x.split(" ")
            score = score.replace("\n", "")

            # name = k
            # score = v
        scr.append((name, score))


    scr = np.array(scr)
    np_scores = scr[:,1].astype(np.float64)
    t = torch.from_numpy(np_scores)


    prob = (torch.sigmoid(t)).data.cpu().numpy()
    prob = np.vstack((np.ones(prob.shape) - prob, prob)).T
    pred = np.argmax(prob, axis=1)

    with open(scores_out, "w") as fh:
        for f, cm, p, pr in zip(scr[:,0],np_scores, pred, prob):
            fh.write('{} {} {} {}\n'.format(f, cm, p, pr))

def join_scores_vote():
    scores = ''
    lines = []
    with open(scores, "r") as f:
        lines = f.readlines()

    file_dict = {}
    for x in lines:
        name, score, pred, proba, probb = x.split(" ")
        
        proba = float(proba.replace("[", ""))
        probb = float(probb.replace("]\n", ""))
        score = float(score)
        pred = float(pred)

        file, fnum, type = name.split("-")
        fnum, iteration = fnum.split("_")
        file = "{}-{}-{}".format(file, fnum, type)

        temp_list = file_dict.get(file)
        if temp_list is None:
            temp_list = [(score, pred, proba, probb)]
        else:
            temp_list.append((score, pred, proba, probb))
        
        file_dict[file] = temp_list
    
    files = file_dict.keys()

    with open('', "w") as f:
        for file in files:
            results = file_dict.get(file)

            results = np.array(results)

            score = results[:,1]
            probsa = results[:,2]
            probsb = results[:,3]

            score = np.mean(score)
            pa = np.mean(probsa)
            pb = np.mean(probsb)
            pred = np.argmax([pa, pb])
            out = "{} {} {} [{} {}]\n".format(file, score, pred, pa, pb)
            f.write(out)


def class_report():
    f1 = ""
    f2 = ""
    f3 = ""

    with open(f1, "r") as f:
        lines = f.readlines()
        y_true1 = []
        y_pred1 = []
        for l in lines:
            name, _, pred, _, _ = l.split(" ")
            yt = 1 if "-REAL" in name else 0
            y_true1.append(yt)
            y_pred1.append(int(pred))
    cm1 = classification_report(y_true1, y_pred1)

    with open(f2, "r") as f:
        lines = f.readlines()
        y_true2 = []
        y_pred2 = []
        for l in lines:
            name, _, pred, _, _ = l.split(" ")
            yt = 1 if "-REAL" in name else 0
            y_true2.append(yt)
            y_pred2.append(int(pred))
    cm2 = classification_report(y_true2, y_pred2)

    with open(f3, "r") as f:
        lines = f.readlines()
        y_true3 = []
        y_pred3 = []
        for l in lines:
            name, _, pred, _, _ = l.split(" ")
            yt = 1 if "-REAL" in name else 0
            y_true3.append(yt)
            y_pred3.append(int(pred))
    cm3 = classification_report(y_true3, y_pred3)


    print(f"LFCC-LCNN Results\n")
    print(cm1)
    print(f"\n\nRawNet2 Results\n")
    print(cm2)
    print(f"\n\nLSSL-wav2vec Results\n")
    print(cm3)

    
# class_report()

# change scores = '' to your own scores files
scores = ''
probs_convert(scores)



