from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import bag_of_words as BOW
import numpy as np


def calculate_prior(Y):
    Y_np_arr = np.asarray(Y)

    total_spam = np.count_nonzero(Y_np_arr)
    total_ham = Y_np_arr.size - np.count_nonzero(Y_np_arr)

    prior_spam = total_spam/Y_np_arr.size
    prior_ham = total_ham/Y_np_arr.size

    return prior_ham, prior_spam


def vocab_prob(X):

    # Seprate Ham and Span Training datas

    X_ham = X[0:792]
    X_spam = X[792:]

    # Sum every column ######
    X_ham_word_sum = np.sum(X_ham, axis=0)
    X_spam_word_sum = np.sum(X_spam, axis=0)

    ##### total number of words in ham and span respectively#####

    X_ham_tatal_words = np.sum(X_ham_word_sum)
    X_spam_tatal_words = np.sum(X_spam_word_sum)

    #### Calculate the probabilty of each word in ham and spam respectively given ham and span repectively########

    X_ham_prob = (X_ham_word_sum + 1)
    X_ham_prob = X_ham_prob / (X_ham_tatal_words + 2)
    X_ham_prob_log = np.log(X_ham_prob)

    X_spam_prob = (X_spam_word_sum + 1)
    X_spam_prob = X_spam_prob / (X_spam_tatal_words + 2)
    X_spam_prob_log = np.log(X_spam_prob)

    return X_ham_prob_log, X_spam_prob_log


def apply_naive_bayes(X_test, X_ham_prob_log, X_spam_prob_log):

    #### Multiplying the Test data with the log probablity of each word and summing the rows########
    # Then exponentiating this result and multiplying it with the prior
    Y_pridict_ham = prior["ham"] * \
        np.exp(np.sum(np.multiply(X_test, X_ham_prob_log), axis=1))

    Y_pridict_spam = prior["spam"] * \
        np.exp(np.sum(np.multiply(X_test, X_spam_prob_log), axis=1))

    # print(Y_pridict_ham)
    # print(Y_pridict_spam)

    Y_pridict = []

    # put 0 if ham and 1 if spam

    for i, j in zip(Y_pridict_ham, Y_pridict_spam):
        if i > j:
            Y_pridict.append(0)
        else:
            Y_pridict.append(1)

    return Y_pridict


##### Execution of the whole code########


####### Creat Training data and Lables    ########

# define file phats
train_datasets_paths, test_datasets_paths = BOW.define_paths()


vocab = BOW.preprocess(datasets=train_datasets_paths)

X = BOW.vactorize(vocab, train_datasets_paths)
Y = BOW.create_y(train_datasets_paths)


######    Create test data and lables   #######

X_test = BOW.vactorize(vocab, test_datasets_paths)
Y_test = BOW.create_y(test_datasets_paths)


####### The naive Bayes Algorithm ##########

prior = {
    "ham": None,
    "spam": None
}


prior["ham"], prior["spam"] = calculate_prior(Y)


X_ham_prob_log, X_spam_prob_log = vocab_prob(X)

Y_pridicted = apply_naive_bayes(
    X_test, X_ham_prob_log, X_spam_prob_log)


accuracy = accuracy_score(Y_test, Y_pridicted)
scores = precision_recall_fscore_support(Y_test, Y_pridicted, average="macro")

print("Model Accuracy :", accuracy, "\n", "Model Precision :",
      scores[0], "\n", "Model Recall :", scores[1], "\n", "Model F1 Score : ", scores[2])
