import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for i, _ in test_set.get_all_Xlengths().items():
        X_test, Xlength_test = test_set.get_item_Xlengths(i)
        scores = {}
        best_word = ""
        best_score = float("-inf")

        for word, model in models.items():
            new_score = float("-inf")
            try:
                new_score = model.score(X_test, Xlength_test)
            except:
                pass

            if new_score >= best_score:
                best_score = new_score
                best_word = word
            scores[word] = new_score

        probabilities.append(scores)
        guesses.append(best_word)
    return probabilities,guesses