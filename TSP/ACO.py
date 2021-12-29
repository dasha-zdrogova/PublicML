import numpy as np
import random as rd
from datetime import datetime

def lengthCal(antPath, distmat):         # Рассчитать расстояние
    length = []
    dis = 0
    for i in range(len(antPath)):
        for j in range(len(antPath[i]) - 1):
            dis += distmat[antPath[i][j]][antPath[i][j + 1]]
        dis += distmat[antPath[i][-1]][antPath[i][0]]
        length.append(dis)
        dis = 0
    return length

def readFile(path):                 # Прочитать граф из файла. Граф задан координатами
    points = []
    with open(path) as f:
        size = int(f.readline())
        distmat = np.array([[0 for _ in range(size)] for _ in range(size)])
        for j in range(size):
            x, y = map(int, f.readline().split())
            points.append([x, y])
            for i in range(len(points) - 1):
                distmat[i][j] = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5
                distmat[j][i] = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5
    return points, distmat, size

def createPoints():                    # Сгенерировать случайные координаты графа
    print('Print graph size')
    size = int(input())
    points = []
    distmat = np.array([[0 for _ in range(size)] for _ in range(size)])
    for j in range(size):
        points.append([rd.randint(0, 100), rd.randint(0, 100)])
        for i in range(len(points) - 1):
            distmat[i][j] = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5
            distmat[j][i] = ((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5
    return points, distmat, size


def ACO(file, antNum=20, alpha=1, beta=3, pheEvaRate=0.3, itermax=500):
    points, distmat, size = readFile(file)
    cityNum = distmat.shape[0]
    pheromone = np.ones((cityNum, cityNum))                   # Феромоновая матрица
    heuristic = 1 / (np.eye(cityNum) + distmat) - np.eye(cityNum)       # Матрица эвристической информации

    startAll = datetime.now()

    for iter in range(itermax):
        antPath = np.zeros((antNum, cityNum)).astype(int) - 1   # Путь каждого муравья
        firstCity = [i for i in range(antNum)]
        rd.shuffle(firstCity)                                   # Перемешивания списка городов
                                                                # которые станут точками старта муравьев

        for i in range(len(antPath)):                           # Начальный город для каждого муравья
            antPath[i][0] = firstCity[i]

        for i in range(cityNum - 1):                            # Обновление пути муравья
            for j in range(antNum):
                unvisted = []                                   # Города, в которых никого не было
                p = []                                          # Вероятности отправиться в город, связанный с тем,
                                                                # в котором находится данный муравей
                pAccum = 0                                      # знаменатель в формуле верятности перехода в город

                for k in range(cityNum):                        # Считывание городов, в которые можно прийти
                    if k not in antPath[j]:
                        unvisted.append(k)

                for m in unvisted:                              # Расчет знаменателя
                    pAccum += pheromone[antPath[j][i]][m] ** alpha * heuristic[antPath[j][i]][m] ** beta

                for n in unvisted:                              # Заполнение списка вероятностей
                    p.append(pheromone[antPath[j][i]][n] ** alpha * heuristic[antPath[j][i]][n] ** beta / pAccum)

                roulette = np.array(p).cumsum()
                r = rd.uniform(min(roulette), max(roulette))
                for x in range(len(roulette)):
                    if roulette[x] >= r:                      # Случайный вобор следеющего города в зависимости от его вероятности
                        antPath[j][i + 1] = unvisted[x]
                        break
        pheromone = (1 - pheEvaRate) * pheromone            # Испарение феромона везде
        length = lengthCal(antPath, distmat)
        for i in range(len(antPath)):
            for j in range(len(antPath[i]) - 1):
                pheromone[antPath[i][j]][antPath[i][j + 1]] += 1 / length[i]     # Добаление феромона там, где прошли
            pheromone[antPath[i][-1]][antPath[i][0]] += 1 / length[i]
    endAll = datetime.now()
    return [(endAll - startAll).total_seconds(), min(length)]
    print(min(length))
    print('minimum path')
    print(antPath[length.index(min(length))])

if __name__ == "__main__":
    print(ACO('test1.txt', itermax=4))