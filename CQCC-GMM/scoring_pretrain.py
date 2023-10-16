from gmm import scoring


# db_list=["LJ_Audio", "FakeAVCeleb"]
db_list = ['WaveFake', 'FakeAVCeleb']

for db in db_list:
# scores file to write
    scores_file = '/CQCC-GMM/scores/{}/_cqccgmm-scores-{}.score'.format(db,db)
    # configs
    features = 'cqcc'
    dict_file = '/CQCC-GMM/final/gmm_cqcc_train_{}.pkl'.format(db)


    db_folder = 'data/{}'.format(db)  # put your database root path here

    eval_folder = db_folder + '/'
    eval_ndx = '/data/{}/lsts/lst.lst'.format(db)

    audio_ext = '.wav'

        # run on ASVspoof 2021 evaluation set
    scoring(scores_file=scores_file, dict_file=dict_file, features=features,
            eval_ndx=eval_ndx, eval_folder=eval_folder, audio_ext=audio_ext,
            features_cached=False)


