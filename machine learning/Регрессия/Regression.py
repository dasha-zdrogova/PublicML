import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

# Генериуем многочлен x^4 + 11x^3 + 9x^2 + 11x + 10000000000
X = np.linspace(-10, 10, 500)
Y = np.array([i**4 - 3*i**3 - i**2 + 7*i + 10000000000 for i in range(len(X))])

# Визулизация данных
fig, ax = plt.figure(), plt.axes()
ax.set_facecolor('#E8E8E8')
fig.patch.set_facecolor('#E8E8E8')
plt.scatter(X, Y, c='#756c83', s=15)
plt.show()

# Добавляем шум

for i in range(500):
    randomNum = np.random.randint(-100000000, 100000000)
    Y[i] = Y[i] + randomNum

# Разделяем данные
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.3)
X = np.array(X).reshape((-1, 1))
Y = np.array(Y).reshape((-1, 1))
X_train = np.array(X_train).reshape((-1, 1))
X_test = np.array(X_test).reshape((-1, 1))
Y_train = np.array(Y_train).reshape((-1, 1))
Y_test = np.array(Y_test).reshape((-1, 1))

# Визуализация всех данных с шумом
fig, ax = plt.figure(), plt.axes()
ax.set_facecolor('#E8E8E8')
fig.patch.set_facecolor('#E8E8E8')
plt.scatter(X, Y, c='#756c83', s=15)
plt.show()

# Пробуем линейную регрессию
lin_reg = LinearRegression().fit(X_train, Y_train)
lin_prediction = lin_reg.predict(X)

lin_see = r2_score(Y, lin_prediction)

fig, ax = plt.figure(), plt.axes()
ax.set_facecolor('#E8E8E8')
fig.patch.set_facecolor('#E8E8E8')
plt.scatter(X_train, Y_train, s=15, c='#756c83')
plt.scatter(X, lin_prediction, s=15, c='#e32636')
plt.xlabel(str(lin_see))
plt.show()

# Пробуем полиномиальную регрессию степени 2
quad_reg = PolynomialFeatures(degree=2)
X_poly = quad_reg.fit_transform(X_train)

lin_for_quad = LinearRegression()
lin_for_quad.fit(X_poly, Y_train)

quad_prediction = lin_for_quad.predict(quad_reg.fit_transform(X))
quad_see = r2_score(Y, quad_prediction)

fig, ax = plt.figure(), plt.axes()
ax.set_facecolor('#E8E8E8')
fig.patch.set_facecolor('#E8E8E8')
plt.scatter(X_train, Y_train, s=15, c='#756c83')
plt.scatter(X, quad_prediction, s=15, c='#e32636')
plt.xlabel(str(quad_see))
plt.show()

# Пробуем полиномиальную регрессию степени 3
cubic_reg = PolynomialFeatures(degree=3)
X_poly = cubic_reg.fit_transform(X_train)

lin_for_cubic = LinearRegression()
lin_for_cubic.fit(X_poly, Y_train)

cubic_prediction = lin_for_cubic.predict(cubic_reg.fit_transform(X))
cubic_see = r2_score(Y, cubic_prediction)

fig, ax = plt.figure(), plt.axes()
ax.set_facecolor('#E8E8E8')
fig.patch.set_facecolor('#E8E8E8')
plt.scatter(X_train, Y_train, s=15, c='#756c83')
plt.scatter(X, cubic_prediction, s=15, c='#e32636')
plt.xlabel(str(cubic_see))
plt.show()

# Пробуем полиномиальную регрессию степени 4
fourth_reg = PolynomialFeatures(degree=4)
X_poly = fourth_reg.fit_transform(X_train)

lin_for_fourth = LinearRegression()
lin_for_fourth.fit(X_poly, Y_train)

fourth_prediction = lin_for_fourth.predict(fourth_reg.fit_transform(X))
fourth_see = r2_score(Y, fourth_prediction)

fig, ax = plt.figure(), plt.axes()
ax.set_facecolor('#E8E8E8')
fig.patch.set_facecolor('#E8E8E8')
plt.scatter(X_train, Y_train, s=15, c='#756c83')
plt.scatter(X, fourth_prediction, s=15, c='#e32636')
plt.xlabel(str(fourth_see))
plt.show()

# Пробуем экспоненциальную регрессию
transformer = FunctionTransformer(np.log, validate=True)
Y_trans = transformer.fit_transform(Y_test)

lin_for_exp = LinearRegression()
result = lin_for_exp.fit(X_test, Y_trans)
Y_fit = result.predict(X)
exp_see = r2_score(Y, Y_fit)

fig, ax = plt.figure(), plt.axes()
ax.set_facecolor('#E8E8E8')
fig.patch.set_facecolor('#E8E8E8')
plt.scatter(X_train, Y_train, s=15, c='#756c83')
plt.scatter(X, np.exp(Y_fit), s=15, c='#e32636')
plt.xlabel(str(exp_see))
plt.show()