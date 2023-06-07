import os, json, pickle

video_root_path_dict = {}

for f1 in ['BAYLORGC{}'.format(n) for n in [218]]:
    video_root_path_dict[f1] = 'D:\\avi_videos\\frames'

for f1 in ['BAYLORCW{}'.format(n) for n in [21]]:
    video_root_path_dict[f1] = 'D:\\avi_videos\\frames'


def main(sess_type):
    save_path = 'make_f1_f2_list_results'

    print('')
    print('sess_type: {}'.format(sess_type))
    print('begins..')
    main_helper(save_path, sess_type)
    print('ends!')


def main_helper(save_path, sess_type):
    '''
    Create f1_f2_list, which contains tuples of form (f1,f2)
    Then, check if the total number of sessions is correct (based on my independent count in Evernote "Sessions used in video analysis").

    To decide which sessions are manually or kilosorted, we use filenames in NeuronalData.
    '''

    if sess_type == 'full_delay':
        f1_f2_list = []

        f1_list = ['BAYLORAT{}'.format(n) for n in [44, 51, 52]]

        for f1 in sorted(f1_list):
            # f1 = 'BAYLORGC[#]'
            for f2 in sorted(os.listdir(os.path.join(video_root_path_dict[f1], f1))):
                # f2 = '[yy][mm][dd]'

                # FILTERING SYSTEM
                if os.path.isfile(os.path.join(video_root_path_dict[f1], f1, f2)):
                    continue

                if '(' in f2 or ')' in f2:
                    continue

                # These sessions don't have neural data.
                # We ignored these sessions previously too and they have videos but no split frames.
                if f1 == 'BAYLORAT44' and f2 in ['201014', '201015']:
                    continue

                # Ignore GC87_2019_09_18 that only has video data and no neural data.
                if f1 == 'BAYLORAT51' and f2 != '210522':
                    continue

                if f1 == 'BAYLORAT52' and f2 in ['210604', '210608', '210603', '210521']:
                    continue

                f1_f2_list.append((f1, f2))

        print('len(f1_f2_list)', len(f1_f2_list))
        prev_f1 = None
        for f1, f2 in f1_f2_list:
            if prev_f1 is None or f1 != prev_f1:
                prev_f1 = f1
                print('')
                print('f1: ', f1)

            print(f1, f2)

    elif sess_type == 'left_stim':
            f1_f2_list = []

            f1_list = ['BAYLORAT{}'.format(n) for n in [40, 44, 46, 48, 52]]

            for f1 in sorted(f1_list):
                # f1 = 'BAYLORGC[#]'
                for f2 in sorted(os.listdir(os.path.join(video_root_path_dict[f1], f1))):
                    # f2 = '[yyyy]_[mm]_[dd]'


                    # FILTERING SYSTEM
                    if os.path.isfile(os.path.join(video_root_path_dict[f1], f1, f2)):
                        continue

                    if '(' in f2 or ')' in f2:
                        continue

                    # These sessions don't have neural data.
                    # We ignored these sessions previously too and they have videos but no split frames.
                    if f1 == 'BAYLORAT40' and f2 == '201023':
                        continue

                    if f1 == 'BAYLORAT46' and f2 == '201128':
                        continue

                    if f1 == 'BAYLORAT48' and f2 == '201130':
                        continue


                    f1_f2_list.append((f1, f2))

            print('len(f1_f2_list)', len(f1_f2_list))
            prev_f1 = None
            for f1, f2 in f1_f2_list:
                if prev_f1 is None or f1 != prev_f1:
                    prev_f1 = f1
                    print('')
                    print('f1: ', f1)

                print(f1, f2)

    elif sess_type == 'early_delay':
        f1_f2_list = []

        f1_list = ['BAYLORAT{}'.format(n) for n in [40, 46, 48, 53, 54]]

        for f1 in sorted(f1_list):
            # f1 = 'BAYLORGC[#]'
            for f2 in sorted(os.listdir(os.path.join(video_root_path_dict[f1], f1))):
                # f2 = '[yyyy]_[mm]_[dd]'


                # FILTERING SYSTEM
                if os.path.isfile(os.path.join(video_root_path_dict[f1], f1, f2)):
                    continue

                if '(' in f2 or ')' in f2:
                    continue

                # These sessions don't have neural data.
                # We ignored these sessions previously too and they have videos but no split frames.
                if f1 == 'BAYLORGC26' and f2 not in ['2018_10_24', '2018_10_25', '2018_10_26']:
                    continue

                # Ignore GC87_2019_09_18 that only has video data and no neural data.
                if f1 == 'BAYLORGC87' and f2 == '2019_09_18':
                    continue

                f1_f2_list.append((f1, f2))

        print('len(f1_f2_list)', len(f1_f2_list))
        prev_f1 = None
        for f1, f2 in f1_f2_list:
            if prev_f1 is None or f1 != prev_f1:
                prev_f1 = f1
                print('')
                print('f1: ', f1)

            print(f1, f2)

        # assert len(f1_f2_list) == 19
        
    elif sess_type == 'all_delay':
        f1_f2_list = []

        f1_list = ['BAYLORAT{}'.format(n) for n in [40, 44, 46, 48, 52]]

        for f1 in sorted(f1_list):
            # f1 = 'BAYLORGC[#]'
            for f2 in sorted(os.listdir(os.path.join(video_root_path_dict[f1], f1))):
                # f2 = '[yyyy]_[mm]_[dd]'


                # FILTERING SYSTEM
                if os.path.isfile(os.path.join(video_root_path_dict[f1], f1, f2)):
                    continue

                if '(' in f2 or ')' in f2:
                    continue

                # These sessions don't have neural data.
                # We ignored these sessions previously too and they have videos but no split frames.
                #if f1 == 'BAYLORAT40' and f2 == '201023':
                #    continue

                #if f1 == 'BAYLORAT46' and f2 == '201128':
                #    continue

                #if f1 == 'BAYLORAT48' and f2 == '201130':
                #    continue

                f1_f2_list.append((f1, f2))

        print('len(f1_f2_list)', len(f1_f2_list))
        prev_f1 = None
        for f1, f2 in f1_f2_list:
            if prev_f1 is None or f1 != prev_f1:
                prev_f1 = f1
                print('')
                print('f1: ', f1)

            print(f1, f2)

    elif sess_type == 'states':
        f1_f2_list = []

        f1_list = ['BAYLORGC{}'.format(n) for n in [218]]

        for f1 in sorted(f1_list):
            # f1 = 'BAYLORGC[#]'
            for f2 in sorted(os.listdir(os.path.join(video_root_path_dict[f1], f1))):
                # f2 = '[yyyy]_[mm]_[dd]'


                # FILTERING SYSTEM
                if os.path.isfile(os.path.join(video_root_path_dict[f1], f1, f2)):
                    continue

                if '(' in f2 or ')' in f2:
                    continue



                f1_f2_list.append((f1, f2))

        print('len(f1_f2_list)', len(f1_f2_list))
        prev_f1 = None
        for f1, f2 in f1_f2_list:
            if prev_f1 is None or f1 != prev_f1:
                prev_f1 = f1
                print('')
                print('f1: ', f1)

            print(f1, f2)

    elif sess_type == '2Pfull':
        
        f1_f2_list = []

        f1_list = ['BAYLORCW{}'.format(n) for n in [21]]

        for f1 in sorted(f1_list):
            # f1 = 'BAYLORCW[#]'
            for f2 in sorted(os.listdir(os.path.join(video_root_path_dict[f1], f1))):
                # f2 = '[yyyy]_[mm]_[dd]'


                # FILTERING SYSTEM
                if os.path.isfile(os.path.join(video_root_path_dict[f1], f1, f2)):
                    continue

                if '(' in f2 or ')' in f2:
                    continue



                f1_f2_list.append((f1, f2))

        print('len(f1_f2_list)', len(f1_f2_list))
        prev_f1 = None
        for f1, f2 in f1_f2_list:
            if prev_f1 is None or f1 != prev_f1:
                prev_f1 = f1
                print('')
                print('f1: ', f1)

            print(f1, f2)

    elif sess_type == 'standard_manual':
        f1_f2_list = []

        f1_list = ['BAYLORGC{}'.format(n) for n in [25, 26, 50, 72, 75, 86, 87, 93, 95]]

        for f1 in sorted(f1_list):
            # f1 = 'BAYLORGC[#]'
            for f2 in sorted(os.listdir(os.path.join(video_root_path_dict[f1], f1))):
                # f2 = '[yyyy]_[mm]_[dd]'
                if os.path.isfile(os.path.join(video_root_path_dict[f1], f1, f2)):
                    continue

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

                # Find a mat file in NeuronalData that contains both f1 and f2.

                cur_mat_files = []  # We need to account for a few sessions that are sorted in both ways.
                for mat_file in filter(lambda x: x[-4:] == '.mat', os.listdir('NeuronalData')):
                    if all(x in mat_file for x in [f1, f2]):
                        cur_mat_files.append(mat_file)

                if len(cur_mat_files) == 1:
                    cur_mat_file = cur_mat_files[0]
                elif len(cur_mat_files) == 2:
                    cur_mat_file = cur_mat_files[0] if 'kilosort' not in cur_mat_files[0] else cur_mat_files[1]
                else:
                    print('')
                    print('f1: {} / f2: {}'.format(f1, f2))
                    print('len(cur_mat_files): {}'.format(len(cur_mat_files)))

                if 'kilosort' in cur_mat_file:
                    continue

                f1_f2_list.append((f1, f2))

        print('len(f1_f2_list)', len(f1_f2_list))
        prev_f1 = None
        for f1, f2 in f1_f2_list:
            if prev_f1 is None or f1 != prev_f1:
                prev_f1 = f1
                print('')
                print('f1: ', f1)

            print(f1, f2)

        assert len(f1_f2_list) == 17

    else:
        raise ValueError('invalid sess type.')

    # Save
    os.makedirs(os.path.join(save_path, sess_type), exist_ok=True)
    with open(os.path.join(save_path, sess_type, 'f1_f2_list.pkl'), 'wb') as f:
        pickle.dump(f1_f2_list, f)


if __name__ == '__main__':
    main('2Pfull')
    # for sess_type in ['{}_{}'.format(task_type, sort_type) for sort_type in ['all', 'manual', 'kilosort'] for task_type in ['standard', 'full_reverse']]:
    #     main(sess_type)