from nltk.util import pr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import bernoulli_model as BER
import numpy as np


def calculate_prior(Y):
    Y_np_arr = np.asarray(Y)

    total_spam = np.count_nonzero(Y_np_arr)
    total_ham = Y_np_arr.size - np.count_nonzero(Y_np_arr)

    prior_spam = total_spam/Y_np_arr.size
    prior_ham = total_ham/Y_np_arr.size

    return prior_ham, prior_spam


def vocab_class_prob(X):

    # Seprate Ham and Span Training datas

    X_ham = X[0:792]
    X_spam = X[792:]

    # # Count number of 1's in  every column ######

    X_ham_word_prob = (
        (np.count_nonzero(X_ham, axis=0)) + 1) / (len(X_ham) + 2)

    X_spam_word_prob = (
        (np.count_nonzero(X_spam, axis=0)) + 1) / (len(X_spam) + 2)

    return X_ham_word_prob, X_spam_word_prob


def calculate_numerator(X_ham_prob, X_spam_prob):
    Y_pridict_ham_numerator_log = (
        np.log(prior["ham"]) + np.sum(np.log(X_ham_prob)))

    Y_pridict_spam_numerator_log = (
        np.log(prior["spam"]) + np.sum(np.log(X_spam_prob)))

    return Y_pridict_ham_numerator_log, Y_pridict_spam_numerator_log


def calculate_denominator(X_test):

    X_prob_one = np.count_nonzero(X_test, axis=0) / len(X_test)
    X_prob_zero = (
        len(X_test) - (np.count_nonzero(X_test, axis=0))) / len(X_test)

    X_test_one = X_test * X_prob_one
    X_test_zero = 1 - X_test * X_prob_zero

    X_deno = X_test_one + X_test_zero

    return np.log(X_deno)


def predict(p_ham, p_spam, X_deno):

    Y_pridict_ham = np.exp(p_ham) * np.exp(np.sum(X_deno, axis=1))

    Y_pridict_spam = np.exp(p_spam) * np.exp(np.sum(X_deno, axis=1))

    Y_pridict = []

    # put 0 if ham is grater and 1 if spam

    for i, j in zip(Y_pridict_ham, Y_pridict_spam):
        if i < j:
            Y_pridict.append(1)
        else:
            Y_pridict.append(0)

    return Y_pridict

    ############################### Execution of the whole code###############################
####### Creat Training data and Lables    ########


train_datasets_paths, test_datasets__paths = BER.define_paths()

####### Creat Training data and Lables    ########
vocab = BER.preprocess(datasets=train_datasets_paths)

X_intermidiate_bow_form, new_vocab = BER.vactorize(vocab, train_datasets_paths)

X = BER.to_bernoulli(X_intermidiate_bow_form, new_vocab)

Y = BER.create_y(train_datasets_paths)


######    Create test data and lables   #######


test_vocab = BER.preprocess(datasets=test_datasets__paths)

X_test_intermidiate_bow_form, new_test_vocab = BER.vactorize(
    vocab, test_datasets__paths)

X_test = BER.to_bernoulli(X_test_intermidiate_bow_form, new_vocab)

Y_test = BER.create_y(test_datasets__paths)


####### The naive Bayes Algorithm ##########

prior = {
    "ham": None,
    "spam": None
}


prior["ham"], prior["spam"] = calculate_prior(Y)


X_ham_prob, X_spam_prob = vocab_class_prob(X)

Y_pridict_ham_numerator_log, Y_pridict_spam_numerator_log = calculate_numerator(
    X_ham_prob, X_spam_prob)


X_deno = calculate_denominator(X_test)


Y_pridicted = predict(Y_pridict_ham_numerator_log,
                      Y_pridict_spam_numerator_log, X_deno)


accuracy = accuracy_score(Y_test, Y_pridicted)
scores = precision_recall_fscore_support(Y_test, Y_pridicted, average="macro")

print("Model Accuracy :", accuracy, "\n", "Model Precision :",
      scores[0], "\n", "Model Recall :", scores[1], "\n", "Model F1 Score : ", scores[2])
