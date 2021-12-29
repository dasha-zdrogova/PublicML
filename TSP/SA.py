import random
import math
from datetime import datetime
import matplotlib.pyplot as plt

# функция генерации координат городов
def read_nodes(path):
    rez = []
    with open(path, 'r', encoding='UTF-8') as f:
        n = int(f.readline())
        for _ in range(n):
            line = f.readline().split()
            rez.append((int(line[0]), int(line[1])))
    return rez

# функция мутации - выбираем 2 случайных узла на пути
# и меняем местами все, что между ними.
def mutate(path):
    i = random.randrange(len(path))
    j = random.randrange(len(path))
    if i > j:
        i, j = j, i
    new_path = path[:i] + path[i:j][::-1] + path[j:]
    return new_path

def run_annealing(initial_path, trials, initial_temp, cooling_rate):
    history = [distance(initial_path)]
    path = initial_path[:]
    temperature = initial_temp
    for trial in range(trials): # для каждой итерации
        new_path = mutate(path)
        delta = (distance(new_path) - distance(path)) / distance(path) # высчитываем разницу пройденного расстояния для пути с мутацией и без
        try:
            if math.exp((delta*(-1))/temperature) > random.random(): # если значение оценки больше случайного числа, то принимаем за текущий новый путь
                path = new_path
        except:
            pass
        temperature *= cooling_rate # уменьшаем температуру
        history.append(distance(path))
    return history, path

# функция для подсчета пройденного расстояния между городами
def distance(path):
    dist = 0
    for i in range(len(path) - 1):
        dist += math.sqrt(math.pow(path[i][0] - path[i+1][0], 2) + math.pow(path[i][1] - path[i+1][1], 2))
    dist += math.sqrt(math.pow(path[0][0] - path[-1][0], 2) + math.pow(path[0][1] - path[-1][1], 2))
    return dist

def SA(file, initial_temp=0.05, cooling_constant=99997, number_of_iteration=100000):
    start = datetime.now()

    path = read_nodes(file)
    history, path = run_annealing(path, number_of_iteration, initial_temp, cooling_constant)

    end = datetime.now()

    return [(end - start).total_seconds(), distance(path)]

    path.append(path[0])
    x = [path[i][0] for i in range(len(path))]
    y = [path[i][1] for i in range(len(path))]

    '''
    plt.figure(1, figsize=(7, 15))
    plt.subplot(211)
    plt.title('Learning curve\n'
              f'Initial temperature: {initial_temp}\n'
              f'Number of trials: {number_of_iteration}\n'
              f'Cooling rate: {cooling_constant}\n'
              f'Found solution: {history[-1]}\n'
              f'Working time: {end - start}', fontsize=10, fontweight='bold')
    plt.xlabel('trials', labelpad=1)
    plt.ylabel('optimal path length', labelpad=2)
    plt.plot(history)
    plt.subplot(212)
    plt.title('Optimal path', pad=2, fontsize=10, fontweight='bold')
    plt.plot(x, y, markersize=12)
    plt.savefig('SA.png', format='png', dpi=300)
    '''
    # plt.show()

if __name__ == "__main__":
    print(SA('test5.txt', number_of_iteration=10000))