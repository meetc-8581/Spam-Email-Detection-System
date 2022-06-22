from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import bag_of_words as BOW
import bernoulli_model as BER


# define data set file paths

train_datasets_paths, test_datasets_paths = BOW.define_paths()


sgd = SGDClassifier()

# Declaring the dictionary containg the parameters for the SGDClassifier.
parameters = {"loss": ["log"], "penalty": ["l2"],
              "max_iter": [300], "learning_rate": ['constant'], "eta0": [0.001]}

# Using the GridSearchCV to fit the SGDClassifier on the training data and to tune the parameters.
model = GridSearchCV(sgd, param_grid=parameters)


# Load the data form Bag of word model

####### Creat Training data and Lables for Bag of words implementation  ########

print("Implementation with the use Bag of Words model data")

vocab = BOW.preprocess(datasets=train_datasets_paths)
X_train_bow = BOW.vactorize(vocab, train_datasets_paths)
Y_train_bow = BOW.create_y(train_datasets_paths)

######    Create test data and lables for Bag of words implementation    #######

_ = BOW.preprocess(datasets=test_datasets_paths)
X_test_bow = BOW.vactorize(vocab, test_datasets_paths)
Y_test_bow = BOW.create_y(test_datasets_paths)


# Training the model on bag of words model, predictng the value on test data and calculating the accuracy.
model.fit(X_train_bow, Y_train_bow)
Y_predicted_bow = model.predict(X_test_bow)


accuracy_bow = accuracy_score(Y_test_bow, Y_predicted_bow)
scores_bow = precision_recall_fscore_support(
    Y_test_bow, Y_predicted_bow, average="macro")

print("Model Accuracy :", accuracy_bow, "\n", "Model Precision :",
      scores_bow[0], "\n", "Model Recall :", scores_bow[1], "\n", "Model F1 Score : ", scores_bow[2])


# Load the data form Bernoulli model

####### Creat Training data and Lables for Bernoulli implementation  ########

print("Implementation with the use Bernoulli model data")


vocab = BER.preprocess(datasets=train_datasets_paths)
X_intermidiate_bow_form, new_vocab = BER.vactorize(vocab, train_datasets_paths)
X_train_bernoulli = BER.to_bernoulli(X_intermidiate_bow_form, new_vocab)
Y_train_bernoulli = BER.create_y(train_datasets_paths)

######    Create test data and lables for Bernoulli implementation    #######

_ = BER.preprocess(datasets=test_datasets_paths)
X_test_intermidiate_bow_form, new_test_vocab = BER.vactorize(
    vocab, test_datasets_paths)
X_test_bernoulli = BER.to_bernoulli(X_test_intermidiate_bow_form, new_vocab)
Y_test_bernoulli = BER.create_y(test_datasets_paths)


# Training the model on Bernoulli model, predictng the value on test data and calculating the accuracy.
model.fit(X_train_bernoulli, Y_train_bernoulli)
Y_predicted_bernoulli = model.predict(X_test_bernoulli)

accuracy_ber = accuracy_score(Y_test_bernoulli, Y_predicted_bernoulli)
scores_bernoulli = precision_recall_fscore_support(
    Y_test_bernoulli, Y_predicted_bernoulli, average="macro")

print("Model Accuracy :", accuracy_ber, "\n", "Model Precision :",
      scores_bernoulli[0], "\n", "Model Recall :", scores_bernoulli[1], "\n", "Model F1 Score : ", scores_bernoulli[2])
