import os, pickle
import numpy as np

def main():
    root_txt_path_list = ['D:\\avi_videos\\frames']

    n_frames_dict = {} # n_frames_dict[(f1,f2)] = n_frmaes

    for root_txt_path in root_txt_path_list:
        print('')
        print('root_path: ', root_txt_path)
        for f1 in sorted(os.listdir(root_txt_path), key=lambda x:int(x[8:])):
            # f1 = BAYLORGC[#]
            for f2 in  sorted(os.listdir(os.path.join(root_txt_path,f1))):
                #f2 = 2019_08_21

                if os.path.isfile(os.path.join(root_txt_path,f1,f2)):
                    continue

                '''
                Standard sessions to ignore
                '''

                if '(' in f2 or ')' in f2:
                    continue

                # These sessions don't have neural data.
                # We ignored these sessions previously too and they have videos but no split frames.
                if f1 == 'BAYLORGC25' and f2 not in ['2018_09_17', '2018_09_21']:
                    continue

                # These sessions don't have neural data.
                # We ignored these sessions previously too and they have videos but no split frames.
                if f1 == 'BAYLORGC26' and f2 not in ['2018_10_24', '2018_10_25', '2018_10_26']:
                    continue

                # Ignore GC87_2019_09_18 that only has video data and no neural data.
                if f1 == 'BAYLORGC87' and f2 == '2019_09_18':
                    continue

                if f1 == 'BAYLORGC86' and f2 == '2019_08_01':
                    continue




                '''
                Full-reverse sessions to ignore
                '''
                # Ignore GC85_2019_08_03 that only has video data and no neural data.
                if f1 == 'BAYLORGC85' and f2 == '2019_08_03':
                    continue

                # Ignore GC88_2019_08_28 that only has video data and no neural data.
                if f1 == 'BAYLORGC88' and f2 == '2019_08_28':
                    continue

                # Ignore because no side view.
                if f1 == 'BAYLORGC90' and f2 == '2019_11_18':
                    continue

                n_frames_dict[(f1,f2)] = main_helper(root_txt_path, f1, f2)

    # Save dict
    save_path = 'get_n_frames_for_each_sess_results'
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'n_frames_dict.pkl'), 'wb') as f:
        pickle.dump(n_frames_dict, f)

    with open(os.path.join(save_path, 'n_frames_dict.pkl'), 'rb') as f:
        n_frames_dict = pickle.load(f)    

    for k, v in n_frames_dict.items():
        print('')
        print(k)
        print('n_frames: ', v)



def main_helper(root_txt_path, f1, f2):
    print('')
    print('f1: {} / f2: {}'.format(f1, f2))

    # def bottom_filter_fct(txt_file):
    #     # txt_file: GC83_20190821_side_trial_264.txt
    #     return txt_file[-4:] == '.txt' and 'passive' not in txt_file and 'bottom' in txt_file
       
    # def side_filter_fct(txt_file):
    #     # txt_file: GC83_20190821_side_trial_264.txt
    #     return txt_file[-4:] == '.txt' and 'passive' not in txt_file and 'side' in txt_file


    def filter_fct(txt_file):
        # txt_file: GC83_20190821_side_trial_264.txt
        return txt_file[-4:] == '.txt' and 'passive' not in txt_file and txt_file[:2] != '._'



    def sort_fct(txt_file):
        # txt_file: GC83_20190821_side_trial_264.txt
        start = txt_file.find('trial')+6
        end = txt_file.find('.txt')
        trial_idx = txt_file[start:end]

        return int(trial_idx)

    n_frames = None
    for txt_file in sorted(filter(filter_fct, os.listdir(os.path.join(root_txt_path, f1, f2))), key=sort_fct): # CW: removed camera file
        # txt_file: GC83_20190821_side_trial_264.txt
        count = 0
        with open(os.path.join(root_txt_path, f1, f2, txt_file), 'r') as f: # CW: removed camera file
            for line in f:
                count += 1
        n_frames = count

        '''
        Since we already confirmed that n_frames is unique in a given session, up to +-5 frames error, 
        and up to a couple outlier trials that have one or so frame, we only need to check one non-outlier trial.
        '''
        if n_frames > 900:
            if np.abs(n_frames - 1000) < np.abs(n_frames - 1200):
                n_frames = 1000
            else:
                n_frames = 1200

        break

    return n_frames




if __name__ == '__main__':
    main()