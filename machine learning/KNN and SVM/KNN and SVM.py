import random
import time
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# Генерация плоских данных (для обоих алгоритмов)
def generateData (numberOfClasses, numberItemsInClass):
    data = []
    for classNum in range(numberOfClasses):
        # Выбираем две случайные точки, которые станут центром для каждого класса
        centerX, centerY = random.random()*60.0, random.random()*60.0

        # Выбираем для каждого узла класса координаты
        # Чтобы точки одного класса находились близко друг к другу,
        # генерируем их координаты с помощью Гауссового распределения
        for rowNum in range(numberItemsInClass):
            data.append([ [random.gauss(centerX,4), random.gauss(centerY,4)], classNum ])
    return data


def showData(nOfClasses, nItemsInClass):
    Colormap = ListedColormap(['#a36c22', '#93b0d0', '#6f2205'])  # Цвета для тренировочных данных

    # Генерируем плоские данные
    df = generateData(nOfClasses, nItemsInClass)
    X = [df[i][0] for i in range(len(df))]
    Y = [df[i][1] for i in range(len(df))]
    # Разделяем данные на тестовые и тренировочные
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Обучаем алгорим на тренировочных данных
    start_learning_timeSVM = time.time()
    # Обучение SVM
    classifierSVM = SVC()
    classifierSVM.fit(X_train, Y_train)
    # Вычисление времени на обучение SVM
    learning_timeSVM = time.time() - start_learning_timeSVM

    start_learning_timeKNN = time.time()
    # Обучение KNN
    classifierKNN = KNeighborsClassifier(n_neighbors=4)
    classifierKNN.fit(X_train, Y_train)
    # Вычисление времени на обучение KNN
    learning_timeKNN = time.time() - start_learning_timeKNN

    # Классификация тренировочных данных
    start_time_SVM = time.time()
    # Классификация SVM
    testDataSVM = classifierSVM.predict(X_test)
    # Вычисление общего времени работы SVM
    time_SVM = time.time() - start_time_SVM + learning_timeSVM
    # Вычисление точности SVM
    accuracySVM = accuracy_score(testDataSVM, Y_test)

    start_time_KNN = time.time()
    # Классификация KNN
    testDataKNN = classifierKNN.predict(X_test)
    # Вычисление общего времени работы KNN
    time_KNN = time.time() - start_time_KNN + learning_timeKNN
    # Вычисление точноссти KNN
    accuracyKNN = accuracy_score(testDataKNN, Y_test)

    plt.subplot(121)
    plt.title("SVM " + str(accuracySVM))
    # Отображение тестовых данных для SVM
    plt.scatter([X_test[i][0] for i in range(len(X_test))],
                [X_test[i][1] for i in range(len(X_test))],
                c=[testDataSVM[i] for i in range(len(Y_test))],
                cmap=Colormap,
                s=10)

    # Отображение тренировочных данных для SVM
    plt.scatter([X_train[i][0] for i in range(len(X_train))],
                [X_train[i][1] for i in range(len(X_train))],
                c=[Y_train[i] for i in range(len(Y_train))],
                cmap=Colormap,
                s=10)
    plt.xlabel(time_SVM)

    plt.subplot(122)
    # Отображение тестовых данных для KNN
    plt.title("KNN " + str(accuracyKNN))
    plt.scatter([X_test[i][0] for i in range(len(X_test))],
                [X_test[i][1] for i in range(len(X_test))],
                c=[testDataKNN[i] for i in range(len(Y_test))],
                cmap=Colormap,
                s=10)
    # Отображение тренировочных данных для SVM
    plt.scatter([X_train[i][0] for i in range(len(X_train))],
                [X_train[i][1] for i in range(len(X_train))],
                c=[Y_train[i] for i in range(len(Y_train))],
                cmap=Colormap,
                s=10)
    plt.xlabel(time_KNN)

    plt.show()


showData(3, 700)

