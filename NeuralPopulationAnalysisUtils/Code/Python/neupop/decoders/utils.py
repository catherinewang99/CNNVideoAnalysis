import numpy as np
from .mdc import MDC

def vec_proj(xs,y):
    return (np.dot(xs,y)/np.linalg.norm(y))

def evaluate_CV(xs, label, classifier_cls = MDC, CV = 10):
    '''
    Perform 10-fold cross validation
    '''

    n = label.size
    shuffled_idx = np.random.shuffle(np.arange(n))
    xs_sh = xs[shuffled_idx].squeeze()
    label_sh = label[shuffled_idx].squeeze()
    step = int(n / CV) ### Might be losing some samples
    tot_correct = 0
    tot_test = 0

    for holdout_round, i in enumerate(range(0,n,step)):
        xs_train = np.vstack([xs_sh[0:i,:], xs_sh[i+step:,:]])
        label_train = np.hstack([label_sh[0:i], label_sh[i+step:]])
        xs_test = xs_sh[i:i+step,:]
        label_test = label_sh[i:i+step]

        classifier = classifier_cls(xs_train, label_train)
        test_correct = classifier.evaluate(xs_test, label_test)[1]
        tot_correct += test_correct
        tot_test += np.size(label_test)
    return 1.*tot_correct/tot_test

def evaluate_CV_2(xs1, label1, xs2, label2, ccls1 = MDC, ccls2 = MDC, even_train_trials=True, CV = 10):
    '''
    Perform 10-fold cross validation.
    classifier for xs2 is trained on all of xs2, but tested on the hold-out of xs1.
    '''

    n = label1.size
    shuffled_idx = np.random.shuffle(np.arange(n))
    xs1_sh = xs1[shuffled_idx].squeeze()
    label1_sh = label1[shuffled_idx].squeeze()
    step = int(n / CV) ### Might be losing some samples
    tot_correct_1 = 0
    tot_test_1 = 0
    tot_correct_2 = 0
    tot_test_2 = 0

    if not even_train_trials:
        classifier2 = ccls2(xs2, label2)

    ## Assume that n_train_2 is larger than n_train_1
    #n_train_1 = n - step
    n_train_2 = label2.size
    #n_min_train = min(n_train_1, n_train_2)

    for holdout_round, i in enumerate(range(0,n,step)):
        xs1_train = np.vstack([xs1_sh[0:i,:], xs1_sh[i+step:,:]])
        label1_train = np.hstack([label1_sh[0:i], label1_sh[i+step:]])
        xs1_test = xs1_sh[i:i+step,:]
        label1_test = label1_sh[i:i+step]

        classifier1 = ccls1(xs1_train, label1_train)
        test_correct_1 = classifier1.evaluate(xs1_test, label1_test)[1]
        tot_correct_1 += test_correct_1
        tot_test_1 += np.size(label1_test)

        # Get the number of train samples from 1 and limit train samples from 2
        if even_train_trials:
            sample_idx = np.random.choice(n_train_2, label1_train.size)
            classifier2 = ccls2(xs2[sample_idx], label2[sample_idx])
            
        test_correct_2 = classifier2.evaluate(xs1_test, label1_test)[1]
        tot_correct_2 += test_correct_2
        tot_test_2 += np.size(label1_test)
    return (1.*tot_correct_1/tot_test_1, 1.*tot_correct_2/tot_test_2)
