import glob
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def define_paths():

   # Define paths and make an array for filenames for train sets

    train_datasets = glob.glob("datasets/*/*/train/ham/*.txt")
    train_datasets.extend(glob.glob("datasets/*Data_Set_2/*/*/train/ham/*txt"))
    train_datasets.extend(glob.glob("datasets/Data_Set_3/*/*/train/ham/*.txt"))
    train_datasets.extend(glob.glob("datasets/*/*/train/spam/*.txt"))
    train_datasets.extend(
        glob.glob("datasets/*Data_Set_2/*/*/train/spam/*txt"))
    train_datasets.extend(
        glob.glob("datasets/Data_Set_3/*/*/train/spam/*.txt"))

    # Define paths and make an array for filenames for test sets

    test_datasets = glob.glob("datasets/*/*/test/*/*.txt")
    test_datasets.extend(glob.glob("datasets/*Data_Set_2/*/*/test/*/*txt"))
    test_datasets.extend(glob.glob("datasets/Data_Set_3/*/*/test/*/*.txt"))

    return train_datasets, test_datasets


def create_y(datasets):

    Y = []

    for filename in datasets:
        # print(filename)
        if filename.find('ham') != -1:
            Y.append(0)
        else:
            Y.append(1)
    return Y


def preprocess(datasets):

    stop_words = set(stopwords.words('english'))
    vocab = set(())

    print("Data Preprocessing...")
    # print(datasets)

    for file in datasets:
        f = open(file, errors="ignore")
        for line in f:
            for word in line.split():
                word = re.sub('[^\w]|[_]|[\d]', "", word)
                word = PorterStemmer().stem(word)
                if word not in stop_words:
                    vocab.add(word)
    print("Data preprecessing completed...")
    return vocab


def vactorize(vocab, datasets):
    data_arr = []
    vectorizer = CountVectorizer()
    for filename in datasets:
        f = open(filename, errors="ignore")
        content = f.read()

        data_arr.append(content)

    vectorizer.fit(vocab)
    vac_vocab = vectorizer.get_feature_names()
    vector = vectorizer.transform(data_arr)

    print("Data vactorized...")
    return vector.toarray(), vac_vocab


def to_bernoulli(X, vocab_t):

    df = pd.DataFrame(X, columns=vocab_t)

    df1 = df.astype(bool).astype(int)

    print("Converted to Bernoulli Model")
    return df1.to_numpy()


if __name__ == "__main__":

    print("bernoulli_model - Module one has been run directly")

    train_datasets_paths, test_datasets_paths = define_paths()
    vocab = preprocess(datasets=train_datasets_paths)
    X, new_vocab = vactorize(vocab, train_datasets_paths)

    X_ber = to_bernoulli(X, new_vocab)

    print(X_ber)
    Y = create_y(train_datasets_paths)

    print(Y)
