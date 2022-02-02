import enum
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

        # Find the number of classes belonging to class 1 and class 2
        for i in y:
            if i == 0:
                n_class1_docs += 1
            else:
                n_class2_docs += 1

        # Find the probability of getting class 1 and class 2 using the above information
        prior[0] = n_class1_docs / n_docs
        prior[1] = n_class2_docs/ n_docs
        
        print("Prior")
        print(prior)

        # Likelihood Calculation (Use x array)
        
        # First, find the total number of words within class 1 and class 2        

        # Loop through the number of documents and find likelihood for class 1
        for document in range(len(x)):
            # If it's class 1
            if y[document][0] == 0:

                # Loop through the word frequencies in the document...
                for wordFreq in x[document]:
                    # and calculate the likelihood of each word in relation to all words in class 1
                    likelihood[document][0] = (wordFreq + 1 / n_words)
            else:
                # If it's class 2
                for wordFreq in x[document]:
                    likelihood[document][1] = (wordFreq + 1 / n_words)

        print(likelihood)
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
