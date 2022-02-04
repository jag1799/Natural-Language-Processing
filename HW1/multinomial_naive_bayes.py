import enum
from turtle import shape
import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!
        
        ###########################
        # YOUR CODE HERE
        
    # Prior Calculation

        # Number of docs belonging to each class
        n_class1_docs = 0
        n_class2_docs = 0

        # Find the number of documents belonging to class 1 and class 2
        for i in y:
            if i == 0:
                n_class1_docs += 1
            else:
                n_class2_docs += 1

        # Find the probability of getting class 1 and class 2 using the above information
        prior[0] = n_class1_docs / n_docs
        prior[1] = n_class2_docs / n_docs

    # Likelihood
        bigDocPositive = np.zeros(shape=(n_class1_docs, n_words))
        bigDocNegative = np.zeros(shape=(n_class2_docs, n_words))
        
        class1Match = 0
        # For all documents in x, extract the ones belonging to class 1.  Use y to do this and store the corresponding rows in x into bigDocPositive.
        for class1Counter in range(n_docs):
            if y[class1Counter][0] == 0:
                bigDocPositive[class1Match][:] = x[class1Counter][:]
                class1Match += 1
        
        # Sum each column within bigDocPositive, and assign that to the corresponding column index of class1Count
        class1Count = np.empty(shape=(n_words))
        class1Count = bigDocPositive.sum(axis=0)
        class1Sum = 0
        print(class1Count)
        # Get the sum of the class1Counts list.  This is the denominator of the likelihood for all class 1 word frequencies.
        for i in range(class1Count.shape[0]):
            class1Sum += class1Count[i]
        
        ################################### DO THE SAME FOR CLASS 2 #############################################
        class2Match = 0
        # For all documents in x, extract the ones belonging to class 2.  Use y to do this and store the corresponding rows in x into bigDocNegative.
        for class2Counter in range(n_docs):
            if y[class2Counter][0] == 1:
                bigDocNegative[class2Match][:] = x[class2Counter][:]
                class2Match += 1

        # Sum up all the columns within class 2
        class2Count = np.empty(shape=(n_words))
        class2Count = bigDocNegative.sum(axis=0)
        class2Sum = 0
        print(class2Count)
        for i in range(class2Count.shape[0]):
            class2Sum += class2Count[i]
        
        for i in range(n_classes):
            for j in range(n_words):
                if i == 0:
                    likelihood[j][i] = (class1Count[j] + self.smooth_param / class1Sum + self.smooth_param)
                else:
                    likelihood[j][i] = (class2Count[j] + self.smooth_param / class2Sum + self.smooth_param)
        
        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
