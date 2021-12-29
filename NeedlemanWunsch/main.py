import time

def main(s1, s2):
    # Задаем параметры для алгоритма
    match = 1
    n, m = len(s1), len(s2)
    INF = int(1e5)
    mismatch = -INF # Чтобы адекватно накладвать строки
    gap = -1
    a = [[0 for _ in range(len(s1) + 1)] for _ in range(len(s2) + 1)] # Создаем матрицу для алгоритма
    # a = np.zeros((m + 1, n + 1))
    # Начинаем заполнять матрицу
    # a[:, 0] = [-i for i in range(m + 1)]
    # a[0, :] = [-i for i in range(n + 1)]
    for i in range(len(s2) + 1):
        a[i][0] = -i
    for j in range(len(s1) + 1):
        a[0][j] = -j

    for i in range(1, len(s2) + 1):
        for j in range(1, len(s1) + 1):
            '''if s2[i - 1] == s1[j - 1]:
                a[i][j] = max(a[i - 1][j - 1] + match, a[i - 1][j] + gap, a[i][j - 1] + gap)
            else:
                a[i][j] = max(a[i - 1][j - 1] + mismatch, a[i - 1][j] + gap, a[i][j - 1] + gap)'''
            a[i][j] = max(a[i - 1][j - 1] + (match if s2[i - 1] == s1[j - 1] else mismatch), a[i - 1][j] + gap, a[i][j - 1] + gap)

    # Выводим готовую матрицу
    #for i in a:
    #   print(*i)

    # Находим обратный путь в матрице

    i = m
    j = n
    s1_new = ''
    s2_new = ''
    while i > 0 or j > 0:
        if i == 0:
            s1_new = s1[:j] + s1_new
            s2_new = '-' * (j) + s2_new
            break

        if j == 0:
            s2_new = s2[:i] + s2_new
            s1_new = '-' * (i) + s1_new
            break

        score = a[i][j]
        scoreDiag = a[i - 1][j - 1]
        scoreUp = a[i - 1][j]
        scoreLeft = a[i][j - 1]
        if s1[j - 1] == s2[i - 1] and score == scoreDiag + match:
            s1_new = s1[j - 1] + s1_new
            s2_new = s2[i - 1] + s2_new
            i -= 1
            j -= 1
        elif s1[j - 1] != s2[i - 1] and score == scoreDiag + mismatch:  # Можно это убрать, потому что тут строки не будут выравнваться
                                                                        # Мы сюда не должны попасть из-за значения mismatch, но в алгоритме есть
            s1_new = s1[j - 1] + s1_new
            s2_new = s2[i - 1] + s2_new
            i -= 1
            j -= 1
        elif score == scoreUp + gap:
            s1_new = '-' + s1_new
            s2_new = s2[i - 1] + s2_new
            i -= 1
        elif  score == scoreLeft + gap:
            s2_new = '-' + s2_new
            s1_new = s1[j - 1] + s1_new
            j -= 1

    # Вывод результата
    return s1_new, s2_new


if __name__ == '__main__':
    # Считываем строки
    start = time.time()
    s = open('./tests/1.txt')
    s1, s2 = (s.readline() for _ in range(2))
    s1_new, s2_new = main(s1, s2)
    # print(s1_new, s2_new, sep='\n')
    print(f'Time elapsed: {time.time() - start}')

