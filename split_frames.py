'''
Main differences from split_frames.py
1. We load session list from make_f1_f2_list.py.
'''

import os, time, json, pickle
from subprocess import call
import subprocess
from multiprocessing import Pool

video_root_path_dict = {}

# Full delay mice
#for f1 in ['BAYLORAT{}'.format(n) for n in [40, 46, 48, 53, 54]]:
#    video_root_path_dict[f1] = 'D:\\avi_videos\\frames'

#for f1 in ['BAYLORGC{}'.format(n) for n in [218]]:
#    video_root_path_dict[f1] = 'D:\\avi_videos\\frames'


for f1 in ['BAYLORCW{}'.format(n) for n in [21]]:
    video_root_path_dict[f1] = 'D:\\avi_videos\\frames'


def main():
    # Load configs
    with open('split_frames_configs.json', 'r') as read_file:
        configs = json.load(read_file)

    # Save
    sess_type = configs['sess_type']
    # assert sess_type in ['{}_{}'.format(task_type, sort_type) for sort_type in ['all', 'manual', 'kilosort'] for
    #                      task_type in ['standard', 'full_reverse']]
    with open(os.path.join('make_f1_f2_list_results', sess_type, 'f1_f2_list.pkl'), 'rb') as f:
        f1_f2_list = pickle.load(f)

    for f1, f2 in f1_f2_list:
        # src_sub_dir: BAYLORGC50/2018_12_11
        split_frames(f1, f2, configs)


def split_frames(f1, f2, configs):
    # src_sub_dir: BAYLORAT51/210520

    src_sub_dir = os.path.join(f1, f2)

    call_list = []

    data_dir = os.path.join(video_root_path_dict[f1], src_sub_dir)

    for avi_file in os.listdir(data_dir):
        # avi_file: GC50_20181211_bottom_trial_100-0000.avi
        if avi_file[-4:] != '.avi':
            continue

        # ignore "passive" files. e.g. GC84_passive_20190724_side_trial_106-0000.avi.
        if 'passive' in avi_file:
            continue

        # In some very rare cases, there are files with prefix '._' like "._GC83_20190815_side_trial_28-0000.avi". Avoid them.
        if avi_file[:2] == '._':
            continue
        if f1 == 'BAYLORGC218' and f2 == '211214':
            continue
        if f1 == 'BAYLORGC218' and f2 == '211215':
            continue
        '''
        We take care of exceptional sessions here.
        1. GC50_2018_12_13
        Mismatch at bottom trial 550/551.
        2. GC83_2019_08_15
        Mismatch at side trial 28/29.
        3. GC84_2019_07_24
        Mismatch at side trial 209/210.
        '''
        # Old method for finding trial number
        if f2 == '230427':
            start = avi_file.find('trial') + 6
            end = avi_file.find('-0000.avi')
        elif f2 == '230503':
            start = avi_file.rfind('l_') + 2
            end = avi_file.rfind('_')
        else:
            start = avi_file.rfind('_') + 1
            end = avi_file.find('.avi')

        trial_idx_str = avi_file[start:end]
        trial_idx = int(trial_idx_str)

        view_type = avi_file.split('_')[2]

        if f1 == 'BAYLORGC50' and f2 == '2018_12_13' and view_type == 'bottom':
            if trial_idx == 550:
                continue
            elif trial_idx >= 551:
                trial_idx_str = str(trial_idx - 1)

        elif f1 == 'BAYLORGC83' and f2 == '2019_08_15' and view_type == 'side':
            if trial_idx == 28:
                continue
            elif trial_idx >= 29:
                trial_idx_str = str(trial_idx - 1)


        elif f1 == 'BAYLORGC84' and f2 == '2019_07_24' and view_type == 'side':
            if trial_idx == 209:
                continue
            elif trial_idx >= 210:
                trial_idx_str = str(trial_idx - 1)



        src = os.path.join(data_dir, avi_file)

        # dst_root_dir: /data5/bkang/alm_video_data/E3/frames
        E_num = configs['dst_root_dir'].split('\\')[-2]  # E3

        # dst_trial_dir: E3-BAYLORGC50-2018_12_10-25
        dst_trial_dir = '-'.join([E_num] + src_sub_dir.split('\\') + [trial_idx_str])

        os.makedirs(os.path.join(configs['dst_root_dir'], src_sub_dir, dst_trial_dir), exist_ok=True)

        # '-'.join(dst_trial_dir, view_type, '%05d.jpg': E3-BAYLORGC50-2018_12_10-25-side-01000.jpg
        dst = os.path.join(configs['dst_root_dir'], src_sub_dir, dst_trial_dir,
                           '-'.join([dst_trial_dir, view_type, '%05d.png']))

        subprocess.run(["C:\\ffmpeg\\bin\\ffmpeg.exe", "-i", src, dst])
    #     call_list += [["ffmpeg", "-i", src, dst]]
    #
    #
    # print('Call: ' + str(type(call)))
    # print(call_list)
    # p = Pool(configs['num_proc'])
    # p.map(call, call_list)

    # p.close()


if __name__ == '__main__':
    main()
