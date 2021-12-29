import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Чтение датасета
df = pd.read_csv('c_0000.csv')

# Посмотрим на зависимость скорости координаты от её положения для x, y, z
X = df.iloc[:, [0, 3]]
Y = df.iloc[:, [1, 4]]
Z = df.iloc[:, [2, 5]]

# Построим графики зависимости
plt.subplot(131)
plt.xlabel('X position of stars')
plt.ylabel('Velocity in X axis of stars')
plt.scatter(X.x, X.vx, color="#2e4600", s=4)

plt.subplot(132)
plt.xlabel('Y position of stars')
plt.ylabel('Velocity in Y axis of stars')
plt.scatter(Y.y, Y.vy, color="#7d4427", s=4)

plt.subplot(133)
plt.xlabel('Z position of stars')
plt.ylabel('Velocity in Z axis of stars')
plt.scatter(Z.z, Z.vz, color="#a2c523", s=4)

plt.show()

# Поскольку зависимость скорости координаты от её положения для x, y, z одинакова,
# Имеет смысл класстеризовать данные по 2 двум признакам:
# Положение координаты и её скорость по какой-нибуль из координат
# Для определённости будем рассматривать координату x

# K-means
# В цикле находим инерцию для каждого значения k от 1 до 15 (k - колличество классов)
# После этого строим график зависимости инерции от количества классов
inertia = []
for k in range(1, 15):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    inertia.append(k_means.inertia_)

plt.plot(range(1, 15), inertia)
plt.show()

# По графику видно, что после k = 4 инерция скорость уменьшения инерции изменяется незначительно
# Получаем, что оптимальным вариантом будет k = 4

kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(X)

X.insert(1, "label", clusters)

plt.scatter(X.x[X.label == 0], X.vx[X.label == 0], color="#2e4600", s=4)
plt.scatter(X.x[X.label == 1], X.vx[X.label == 1], color="#7d4427", s=4)
plt.scatter(X.x[X.label == 2], X.vx[X.label == 2], color="#a2c523", s=4)
plt.scatter(X.x[X.label == 3], X.vx[X.label == 3], color="#486b00", s=4)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color="orange", s=200) # scentroidler

plt.show()
