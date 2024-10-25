import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def normalizeData(data):
    X = data.drop(columns=['Activity'])
    y = data['Activity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def KNearestNeighbours(data):
    X_train, X_test, y_train, y_test = normalizeData(data)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("K-NN Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("K-NN Classification Report:\n", classification_report(y_test, y_pred_knn))
    return y_pred_knn

def NaiveBayes(data):
    X_train, X_test, y_train, y_test = normalizeData(data)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
    return y_pred_nb

def SVMClassification(data):
    X_train, X_test, y_train, y_test = normalizeData(data)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
    return y_test, y_pred_svm

def visualizeComparison(y_test, y_pred_knn, y_pred_nb, y_pred_svm):
    accuracies = [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_svm)]
    labels = ['K-NN', 'Naive Bayes', 'SVM']

    accuracies = [accuracy * 100 for accuracy in accuracies]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=['blue', 'purple', 'cyan'])
    plt.title('Porównanie dokładności modeli')
    plt.xlabel('Model')
    plt.ylabel('Dokładność (%)')

    plt.ylim(0, 100)
    plt.gca().set_yticks(range(0, 101, 10))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f'{accuracy:.1f}%',
            ha='center',
            va='bottom'
        )
    plt.show()

def neuralNetwork(data):
    X_train, X_test, y_train, y_test = normalizeData(data)
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y_train_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.2, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
    print(f'Neural Network Test Accuracy: {test_accuracy * 100:.2f}%')

    # loss chart
    plt.plot(history.history['loss'], label='Strata treningowa')
    plt.plot(history.history['val_loss'], label='Strata walidacyjna')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()
    plt.title('Strata w trakcie uczenia')
    plt.show()

    # accuraccy chart
    plt.plot(history.history['accuracy'], label='Dokładność treningowa')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność (%)')
    plt.legend()
    plt.title('Dokładność w trakcie uczenia')
    plt.show()


if __name__ == '__main__':
    newProcessed = pandas.read_csv('./datasets/task1/NewProcessedData.csv')
    originalProcessed = pandas.read_csv('./datasets/task1/OriginalProcessedData.csv')

    y_pred_knn = KNearestNeighbours(newProcessed)
    y_pred_nb = NaiveBayes(newProcessed)
    y_test, y_pred_svm = SVMClassification(newProcessed)

    visualizeComparison(y_test, y_pred_knn, y_pred_nb, y_pred_svm)

    neuralNetwork(newProcessed)