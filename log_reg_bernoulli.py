import bernoulli_model as BER
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from log_reg import model_test, model_train


####### Creat Training data and Lables    ########

train_datasets_paths, test_datasets_paths = BER.define_paths()

########### BAg of Words modle is used for following impleentation########

vocab = BER.preprocess(datasets=train_datasets_paths)

X_intermidiate_bow_form, new_vocab = BER.vactorize(vocab, train_datasets_paths)

X = BER.to_bernoulli(X_intermidiate_bow_form, new_vocab)

Y = BER.create_y(train_datasets_paths)

######    Create test data and lables   #######


_ = BER.preprocess(datasets=test_datasets_paths)

X_test_intermidiate_bow_form, new_test_vocab = BER.vactorize(
    vocab, test_datasets_paths)

X_test = BER.to_bernoulli(X_test_intermidiate_bow_form, new_vocab)

Y_test = BER.create_y(test_datasets_paths)


# Shuffle and Split the training data into training and validation sets   #####

X_train, X_validate, Y_train, Y_validate = train_test_split(
    X, Y, test_size=0.3, random_state=4)


######## Train the model ###########

W0_final, W_final, Log_loss = model_train(X_train, Y_train, lam=1)


# Test the model ############
print("This is  bernoulli Model Implementation")


Y_pridicted = model_test(W0_final, W_final, X_test)

accuracy = accuracy_score(Y_test, Y_pridicted)
scores = precision_recall_fscore_support(Y_test, Y_pridicted, average="macro")

print("Model Accuracy :", accuracy, "\n", "Model Precision :",
      scores[0], "\n", "Model Recall :", scores[1], "\n", "Model F1 Score : ", scores[2])


####### plot the graph of loss####

plt.plot(Log_loss)
plt.ylabel('Loss')
plt.grid(True)
plt.show()
