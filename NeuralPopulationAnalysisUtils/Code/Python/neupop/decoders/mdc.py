import numpy as np
from . import utils

class MDC(object):
    '''
    Mean Difference / Nearest Centroid Classifier

    Binary classifier with class labels 0 and 1.
    MDC uses the difference of means of the two classes as its W, the coding direction.

    Input:
    data: n x d numpy array
    label: length n numpy array of 0's and 1's
    verbose : whether to print details, e.g. training results

    Parameters:
    W : coding direction / weights
    mean : mean of the whole training data set.
    verbose : verbose from input
    '''

    def __init__(self, data, label, verbose=False):
        self.verbose = verbose
        # assume 2 classes
        self.mean = np.empty([]) # For mean subtraction
        self.W = np.empty([])
        self._learn(data, label)

    def _preprocess(self, xs, subtract_mean=True):
        if subtract_mean:
            return (xs - self.mean)
        return xs

    def _learn(self, data, label):
        mask0 = label==0
        mask1 = label==1

        self.mean = np.mean(data, axis=0)
        data_normed = data - self.mean # mean subtraction

        mean0 = np.mean(data_normed[mask0],axis=0)
        mean1 = np.mean(data_normed[mask1],axis=0)

        W = mean1 - mean0
        #b = -(np.power(np.linalg.norm(mean1),2)-np.power(np.linalg.norm(mean0),2))/2
        self.W = W

        if self.verbose:
            # Compute training accuracy
            training_accuracy = self.evaluate(data,label)[0]
            print ("Training accuracy: ", training_accuracy)

    def classify(self, x):
        # mean subtract
        x_preprocessed = self._preprocess(x)
        f = np.dot(self.W.T, x_preprocessed.T)
        #f = np.linalg.norm(np.matmul(W.T, x.T)*x/(np.linalg.norm(x)**2)) + b
        return np.squeeze(f > 0).astype(int)

    def evaluate(self, xs, label):
        predicted_classes = self.classify(xs)
        correct_number = np.sum(predicted_classes==label)
        prediction_accuracy = correct_number / np.size(label)
        #print ("Evaluation accuracy: ", prediction_accuracy)
        return (prediction_accuracy, correct_number)

    def project_to_W(self, xs, subtract_mean=True):
        x_preprocessed = self._preprocess(xs, subtract_mean)
        return (utils.vec_proj(x_preprocessed, self.W))
