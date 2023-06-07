import numpy as np
from .mdc import MDC

class DFDAC(MDC):
    '''
    Diagonal Fisher Discriminant Analysis Classifier

    Binary classifier with class labels 0 and 1.
    DFDAC uses the same W as FDAC, but only the diagonal of the "within" scatter matrix Sw is used in computing the W.

    Features for which diagonal of Sw is 0 are removed.

    Input:
    data: n x d numpy array
    label: length n numpy array of 0's and 1's
    verbose : whether to print details, e.g. training results

    Parameters:
    W : coding direction / weights
    mean : mean of the whole training data set.
    verbose : verbose
    nd : dimensionality of the feature space
    n : training data set size
    useful_features_mask : boolean array indicating which features have non-zero within scatter
    '''

    def __init__(self, data, label, verbose=False):
        self.verbose = verbose
        # assume 2 classes
        self.nd = np.shape(data)[1]
        self.n = np.size(label) #size of training set
        self.mean = np.empty([]) # For mean subtraction
        self.W = np.empty([])
        self.useful_features_mask = np.empty([])
        self._learn(data, label)

    def _preprocess(self, xs, subtract_mean=True):
        # Keep only useful features and fix data structure if necessary
        if np.size(np.shape(xs)) == 1:
            xs_normed = xs[self.useful_features_mask]
            xs_normed = xs_normed.reshape([1, -1])
        else:
            xs_normed = xs[:,self.useful_features_mask]

        if subtract_mean:
            xs_normed = xs_normed - self.mean

        return (xs_normed)

    def _learn(self, data, label):

        mask0 = label==0
        mask1 = label==1

        n0 = np.sum(mask0)
        n1 = np.sum(mask1)
        assert self.n == n0+n1

        # mean subtract
        mean = np.mean(data, axis=0)
        data_normed = data-mean

        data0 = data_normed[mask0]
        data1 = data_normed[mask1]

        mean0 = np.mean(data0,axis=0)
        mean1 = np.mean(data1,axis=0)

        # Scatter matrices
        S0_diag = (n0-1) * np.var(data0,axis=0)
        S1_diag = (n1-1) * np.var(data1,axis=0)
        Sw_diag = S0_diag + S1_diag

        # Only features with any within scatter is useful
        self.useful_features_mask = Sw_diag != 0
        self.mean = mean[self.useful_features_mask]

        # In computing Sw and Sb, only take the useful features
        Sw_inv = np.diag(1/Sw_diag[self.useful_features_mask])
        meandiff = (mean1-mean0)[self.useful_features_mask]
        W = np.matmul(Sw_inv, meandiff)
        self.W = W

        if self.verbose:
            # Compute training accuracy
            training_accuracy = self.evaluate(data,label)[0]
            print ("Training accuracy: ", training_accuracy)
