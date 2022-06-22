import bag_of_words as BOW
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


import json


def predict(W0, W, X):

    degree = W0 + np.sum(np.multiply(W, X), axis=1)
    Y_pred = 1 / (1 + np.exp(-degree))
    return Y_pred


def model_train(X_train, Y_train, lam=1.0):

    W0 = 0.0
    W = np.zeros(len(X_train[0]))
    L = -0.001
    epochs = 300

    Loss_log = []

    print("Model Training....")
    print("Lambda : ", lam, "Epochs: ",  epochs, "learning Rate: ", L)
    print("This may take some time")

    for epoch in range(epochs):

        # print(epoch)

        Y_pred = predict(W0, W, X_train)

        # # calculate loss
        loss_W0 = np.sum(np.subtract(Y_train, Y_pred))
        # print(loss_W0)

        loss = np.dot(np.transpose(X_train),
                      np.subtract(Y_train, Y_pred))
        # print(loss)
        # print(len(loss))

        W0 = W0 - L * (loss_W0 - (lam * W0))
        # print(W0)

        W = np.subtract(
            W, (np.multiply(L, (np.subtract(loss, np.multiply(lam, W)))))
        )

        Loss_log.append(np.sum(loss)/len(loss))
    print("Training Completed...")
    return W0, W, Loss_log

    ##################   CODE EXECUTION    ###################
    # Define paths and make an array for filenames for train sets


def model_test(W0_final, W_final, X):

    print("Model Testing...")

    Y_pred = predict(W0_final, W_final, X)

    Y_pridict = []

    for i in Y_pred:
        if i > 0.5:
            Y_pridict.append(1)
        else:
            Y_pridict.append(0)

    return Y_pridict

#
#
#
#
#


if __name__ == "__main__":

    # define data set file paths

    train_datasets__paths, test_datasets_paths = BOW.define_paths()

    ########### Bag of Words modle is used for following impleentation########

    ####### Creat Training data and Lables    ########

    vocab = BOW.preprocess(datasets=train_datasets__paths)
    X = np.array(BOW.vactorize(vocab, train_datasets__paths))
    Y = np.array(BOW.create_y(train_datasets__paths))

    ######    Create test data and lables   #######

    _ = BOW.preprocess(datasets=test_datasets_paths)
    X_test = BOW.vactorize(vocab, test_datasets_paths)
    Y_test = BOW.create_y(test_datasets_paths)

    # Shuffle and Split the training data into training and validation sets   #####

    X_train, X_validate, Y_train, Y_validate = train_test_split(
        X, Y, test_size=0.3, random_state=4)

    ######## Train the model ###########

    W0_final, W_final, Loss_log = model_train(X_train, Y_train, lam=15)

    # np.savetxt("file1.txt", W_final)


    # Test the model ############
    Y_pridicted = model_test(W0_final, W_final, X_test)

    accuracy = accuracy_score(Y_test, Y_pridicted)

    scores = precision_recall_fscore_support(
        Y_test, Y_pridicted, average="macro")

    print("Model Accuracy :", accuracy, "\n", "Model Precision :",
          scores[0], "\n", "Model Recall :", scores[1], "\n", "Model F1 Score : ", scores[2])

    ####### plot the graph of loss####

    plt.plot(Loss_log)
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
