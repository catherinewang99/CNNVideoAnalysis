import numpy as np
from sklearn import decomposition
from .mdc import MDC

class FDAC(MDC):
    '''
    Fisher Discriminant Analysis Classifier

    Binary classifier with class labels 0 and 1.
    FDAC uses the vector along which distribution of the two classes, upon the data points are projected, are maximized as its W, the coding direction.

    If the dimensionality in the feature space is larger than the size of training set, then PCA is performed to reduce the dimensionality of the feature space down to the training set size. This is an arbitrary choice of dealing with the dimensionality problem.

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
    do_PCA : flag indicating whether PCA will be performed (i.e. rank < nd)
    '''

    def __init__(self, data, label, verbose=False):
        self.verbose = verbose
        # assume 2 classes
        self.nd = np.shape(data)[1]
        self.do_PCA = False
        self.n = np.size(label) #size of training set
        self.mean = np.empty([]) # For mean subtraction
        # if total data rank < self.nd, set up PCA:
        dataRank = np.linalg.matrix_rank(data)
        if dataRank < self.nd:
            if self.verbose:
                print("Using PCA to reduce dimensions")
                print("n=", self.n, ", rank=", dataRank, ", nd=", self.nd)
            self.do_PCA = True
            self.pca = decomposition.PCA(n_components=(dataRank))
        self.W = np.empty([])
        self._learn(data, label)

    def _preprocess(self, xs, subtract_mean=True):
        if np.size(np.shape(xs)) == 1:
            xs = xs.reshape([1, -1])

        if self.do_PCA:
            xs = self.pca.transform(xs)

        if subtract_mean:
            xs = xs - self.mean

        return (xs)

    def _learn(self, data, label):

        mask0 = label==0
        mask1 = label==1

        n0 = np.sum(mask0)
        n1 = np.sum(mask1)
        assert self.n == n0+n1

        # If not full rank, do PCA to reduce dimensionality to n-2
        if self.do_PCA:
            data_reduced = self.pca.fit_transform(data)
        else:
            data_reduced = data

        # mean subtract after PCA
        self.mean = np.mean(data_reduced, axis=0)
        data_normed = data_reduced-self.mean # mean subtraction

        data0 = data_normed[mask0]
        data1 = data_normed[mask1]

        mean0 = np.mean(data0,axis=0)
        mean1 = np.mean(data1,axis=0)

        # Scatter matrices
        S0 = (n0-1) * np.cov(data0, rowvar=False)
        S1 = (n1-1) * np.cov(data1, rowvar=False)
        # Within scatter
        Sw = S0 + S1
        Sw_inv = np.linalg.inv(Sw)
        W = np.matmul(Sw_inv, (mean1-mean0))

        self.W = W

        if self.verbose:
            # Compute training accuracy
            training_accuracy = self.evaluate(data,label)[0]
            print ("Training accuracy: ", training_accuracy)
