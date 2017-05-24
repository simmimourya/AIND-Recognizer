import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None
        num_features = self.X.shape[1]

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            # train a model based on current number of states/components and find it's score

            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                # likelihood log
                logL = hmm_model.score(self.X, self.lengths)

                # log of number of data points
                logN = np.log(len(self.X))
                # As discussed here: https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/4

                # Initial state occupation probabilities = numStates
                # Transition probabilities = numStates*(numStates - 1)
                # Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
                # numMeans and numCovars are the number of means and covars calculated. One mean and covar for each state and features. 
                
                # Then the total number of parameters are:
                # Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities

                occupation_probabilities = num_states
                transition_probabilities = num_states * (num_states - 1)
                emission_probabilities = num_states * num_features * 2
                p = occupation_probabilities + transition_probabilities + emission_probabilities 

                bic_score = -2 * logL + p * logN

                best_score, best_model = max((best_score, best_model), (bic_score, hmm_model))
            except:
                pass
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    # As discussed here: https://discussions.udacity.com/t/dic-criteria-clarification-of-understanding/233161/2
    # we are trying to find the model that gives a high likelihood(small negative number) to the original word
    # and low likelihood(very big negative number) to the other words. 
    
    # So DIC score is
    # DIC = log(P(original world)) - average(log(P(otherwords)))

    # We need to maximize DIC. 

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_DIC_Score = float('-inf')
        M = len((self.words).keys())  # num of words

        best_hmm_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                LogL = hmm_model.score(self.X, self.lengths)

            except:
                LogL = float("-inf")

            SumLogL = 0

            for each_word in self.hwords.keys():
                X_each_word, lengths_each_word = self.hwords[each_word]

            try:
                SumLogL += hmm_model.score(X_each_word, lengths_each_word)

            except:
                SumLogL += 0

            # DIC = log(P(original world)) - average(log(P(otherwords)))
            # SumLogL - LogL effectively means sum_of_logs(P(allwords) - log(P(original world) i.e. sum_of_logs(P(other_words)

            DIC_Score = LogL - (1 / (M - 1)) * (SumLogL - (0 if LogL == float("-inf") else LogL))

            # We need to maximize DIC score

            if DIC_Score > best_DIC_Score:
                best_DIC_Score = DIC_Score
                best_hmm_model = hmm_model

        return best_hmm_model

class SelectorCV(ModelSelector):
    """
        CV technique includes breaking-down the training set into "folds",
        rotating which fold is "left out" of the training set.
        The fold that is "left out" is scored for validation.
        Use this as a proxy method of finding the
        "best" model to use on "unseen data". Higher the CV score the "better" the model.
    """

    def calc_best_score_cv(self, score_cv):
        return max(score_cv, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        kf = KFold(n_splits = 3, shuffle = False, random_state = None)
        log_likelihoods = []
        score_cvs = []

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Check sufficient data to split using KFold
                if len(self.sequences) > 2:
                    # Break down the sequences into folds.
                    for train_index, test_index in kf.split(self.sequences):

                        self.X, self.lengths = combine_sequences(train_index, self.sequences)

                        X_test, lengths_test = combine_sequences(test_index, self.sequences)

                        hmm_model = self.base_model(num_states)
                        log_likelihood = hmm_model.score(X_test, lengths_test)
                else:
                    hmm_model = self.base_model(num_states)
                    log_likelihood = hmm_model.score(self.X, self.lengths)

                log_likelihoods.append(log_likelihood)

                # Find average Log Likelihood of CV fold
                score_cvs_avg = np.mean(log_likelihoods)
                score_cvs.append(tuple([score_cvs_avg, hmm_model]))

            except Exception as e:
                pass
        return self.calc_best_score_cv(score_cvs)[1] if score_cvs else None
