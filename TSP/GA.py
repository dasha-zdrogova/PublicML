import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Вычисление евклидового расстояния
def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance


# Создание первой популяции
def init_population(cityList, populationSize):
    population = []
    for i in range(populationSize):
        population.append(random.sample(cityList, len(cityList)))
    return population


# Расчёт длины всего маршрута
def fit(path):
    distanceSum = 0
    for i in range(len(path) - 1):
        distanceSum += distance(path[i], path[i + 1])
    distanceSum += distance(path[-1], path[0])
    return 1 / distanceSum


# Ранжирование популяции по весам
def rank(population):
    populationRank = []
    for i in range(len(population)):
        populationRank.append((i, fit(population[i])))
    return sorted(populationRank, key=lambda x: x[1])


# Случайный выбор родителя
def roulette_wheel_selection(populationRank):
    probabilitySum = 0
    for i in range(len(populationRank)):
        probabilitySum += populationRank[i][1]
    randomNum = random.uniform(0.0001, probabilitySum)

    i = 0
    while randomNum > 0 and i < len(populationRank):
        randomNum -= populationRank[i][1]
        i += 1
    else:
        return populationRank[i-1][0]


# Скрещивание особей
def breed(populationRank, numOfCity, population):
    newPopulation = []
    for i in range(len(population)):
        parent1 = population[roulette_wheel_selection(populationRank)]
        parent2 = population[roulette_wheel_selection(populationRank)]
        child = []
        if parent1 != parent2:
            sep = random.randint(1, numOfCity-1)
            for i in range(sep):
                child.append(parent1[i])
            for i in range(len(parent2)):
                if parent2[i] not in child:
                    child.append(parent2[i])
        else:
            child = parent1
        newPopulation.append(child)
    return newPopulation


# Мутация
def mutate(population, mutationRate, numOfCity):
    for i in range(len(population)):
        for j in range(len(population[i])):
            randomNum = random.random()
            if randomNum < mutationRate:
                # Будем менять два города в перестановке местами
                a = random.randint(0, len(population[i]) - 1)
                population[i][a], population[i][j] = population[i][j], population[i][a]
    return population


# Создание следующего поколения
def next_population(population, numOfCity, mutationRate):
    populationRank = rank(population)                                   # Вычисляем коэффициенты выживаемости
    breedPopulation = breed(populationRank, numOfCity, population)      # Размножение
    nextGeneration = mutate(breedPopulation, mutationRate, numOfCity)   # Новое поколение
    return nextGeneration


def GA(path, population_size=10, rate=0.01, generations=4000):
    cityList = [] # Список городов
    startAll = datetime.now()
    # Считывание координат городов из файла
    with open(path, 'r', encoding='UTF-8') as f:
        n = int(f.readline())
        for _ in range(n):
            line = f.readline().split()
            cityList.append((int(line[0]), int(line[1])))

    numOfCity = len(cityList) # Количество городов
    populationSize = population_size # Размер популяции
    mutationRate = rate # Вероятность мутации
    generation = generations # Количество поколений
    
    population = init_population(cityList, populationSize)
    process = []

    for i in range(generation):
        Population = next_population(population, numOfCity, mutationRate)
        rankPopulation = rank(Population)
        process.append([1 / rankPopulation[-1][1], Population[rankPopulation[-1][0]]])

    endAll = datetime.now()
    way = min(process)[1]
    return [(endAll - startAll).total_seconds(), 1/fit(way)]

    print('Наилучший маршрут:')
    for i in range(numOfCity):
        if i == numOfCity-1:
            print(way[i])
        else:
            print(way[i], '-->', end='', sep='')

    # Построение графиков
    '''
    plt.subplot(211)
    plt.plot([process[i][0] for i in range(len(process))], linewidth=0.45)
    plt.xlabel('Generation')
    plt.ylabel('Distance')

    plt.subplot(212)
    X = []
    Y = []
    for i in range(len(way)):
        X.append(way[i][0])
        Y.append(way[i][1])
    plt.plot(X, Y, c='b')
    plt.plot([X[0], X[-1]], [Y[0], Y[-1]], c='b')
    plt.scatter(X, Y)
    plt.xlabel('Way')
    plt.savefig('GA.png', format='png', dpi=300)
    plt.show()'''

# Генерация координат городов
'''for i in range(numOfCity):
    cx = random.randint(0, 50)
    cy = random.randint(0, 50)
    cityList.append((cx, cy))'''

if __name__ == "__main__":
    print(GA('test1.txt', generations=600))
