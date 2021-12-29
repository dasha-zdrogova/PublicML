import random
import time
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# Генерация плоских данных (для обоих алгоритмов)
def generateData (numberOfClasses, numberItemsInClass):
    data = []
    for classNum in range(numberOfClasses):
        # Выбираем две случайные точки, которые станут центром для каждого класса
        centerX, centerY = random.random()*6.0, random.random()*6.0

        # Выбираем для каждого узла класса координаты
        # Чтобы точки одного класса находились близко друг к другу,
        # генерируем их координаты с помощью Гауссового распределения
        for rowNum in range(numberItemsInClass):
            data.append([ [random.gauss(centerX,0.3), random.gauss(centerY,0.3)], classNum ])
    return data


def showDataOnMesh(nOfClasses, nItemsInClass):
    trainColormap = ListedColormap(['#a36c22', '#93b0d0', '#6f2205'])  # Цвета для тренировочных данных
    testColormap = ListedColormap(['#eacc97', '#dee8ec', '#bf5d39'])  # Цвета для тестовых данных

    # Генерируем сетку, каждый узел которой будет тестовыми данными
    def generateTestMesh(trainData):
        min_x = min([trainData[i][0][0] for i in range(len(trainData))])
        max_x = max([trainData[i][0][0] for i in range(len(trainData))])
        min_y = min([trainData[i][0][1] for i in range(len(trainData))])
        max_y = max([trainData[i][0][1] for i in range(len(trainData))])
        testX, testY = np.meshgrid(np.arange(min_x - 1.0, max_x + 1.0, 0.05),
                                   np.arange(min_y - 1.0, max_y + 1.0, 0.05))
        return [testX, testY]

    # Все сгенерированные данные являются тренировочными
    trainData = generateData(nOfClasses, nItemsInClass)

    # Обучаем алгорим на тренировочных данных
    start_learning_timeSVM = time.time()
    classifierSVM = SVC()
    classifierSVM.fit([trainData[i][0] for i in range(len(trainData))],
                   [trainData[i][1] for i in range(len(trainData))])
    learning_timeSVM = time.time() - start_learning_timeSVM

    start_learning_timeKNN = time.time()
    classifierKNN = KNeighborsClassifier(n_neighbors=4)
    classifierKNN.fit([trainData[i][0] for i in range(len(trainData))],
                   [trainData[i][1] for i in range(len(trainData))])
    learning_timeKNN = time.time() - start_learning_timeKNN

    # Генерируем тестовые данные
    testMesh = generateTestMesh(trainData)

    # Классификация тренировочных данных
    testMeshLabelsSVM = []
    testMeshLabelsKNN = []

    start_time_SVM = time.time()
    for node in zip(testMesh[0].ravel(), testMesh[1].ravel()):
        testMeshLabelsSVM.append(classifierSVM.predict([node]))
    time_SVM = time.time() - start_time_SVM + learning_timeSVM

    start_time_KNN = time.time()
    for node in zip(testMesh[0].ravel(), testMesh[1].ravel()):
        testMeshLabelsKNN.append(classifierKNN.predict([node]))
    time_KNN = time.time() - start_time_KNN + learning_timeKNN

    plt.subplot(121)
    plt.title("SVM " + str(time_SVM))
    # Отображение тестовых данных для SVM
    plt.pcolormesh(testMesh[0],
                   testMesh[1],
                   np.asarray(testMeshLabelsSVM).reshape(testMesh[0].shape),
                   cmap=testColormap)

    # Отображение тренировочных данных для SVM
    plt.scatter([trainData[i][0][0] for i in range(len(trainData))],
                [trainData[i][0][1] for i in range(len(trainData))],
                c=[trainData[i][1] for i in range(len(trainData))],
                cmap=trainColormap,
                s=20)

    plt.subplot(122)
    # Отображение тестовых данных для KNN
    plt.title("KNN " + str(time_KNN))
    plt.pcolormesh(testMesh[0],
                   testMesh[1],
                   np.asarray(testMeshLabelsKNN).reshape(testMesh[0].shape),
                   cmap=testColormap)

    # Отображение тренировочных данных для KNN
    plt.scatter([trainData[i][0][0] for i in range(len(trainData))],
                [trainData[i][0][1] for i in range(len(trainData))],
                c=[trainData[i][1] for i in range(len(trainData))],
                cmap=trainColormap,
                s=20)

    plt.show()


showDataOnMesh(3, 70)
