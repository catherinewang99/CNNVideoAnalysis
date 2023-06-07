from session import Session

def ed_unit_test():
    data = Session('../dataset_prepared_2014_09_18_ALMTransientInhibition/ANM257772_20141121.mat')

    ## Basic checks
    data.n_units == data.units.shape[0]
    data.n_trials == data.units.shape[1]
    len(data.stim_on_time) == data.n_trials

    ## Changing values in a deepcopy should not affect the original
    data2 = data.deepcopy()
    data2.task_pole_off_time[0] = 10000
    assert data.task_pole_off_time[0] != 10000


    ## Aligning time should change the flag data.time_aligned
    assert data.time_aligned == False
    data.align_time()
    assert data.time_aligned == True

    ## Selecting units should result in filtering the correct units
    data3 = data.deepcopy()
    data3.select_units([3])
    assert data3.n_units == 1
    assert data3.units.shape[0] == 1
    for (a,b) in zip(data.units[3], data3.units[0]):
        assert np.array_equal(a,b)

    print('All tests passed.')

def main():
    ed_unit_test()

if __name__ == '__main__':
    main()
