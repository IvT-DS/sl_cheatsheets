import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

np.random.seed(666)
x = np.arange(0, 10)
y = np.sin(x) + np.random.normal(0, .4, size=len(x))

x_test = np.arange(9, 20)
y_test = np.sin(x_test) + np.random.normal(0, .2, size=len(x_test))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax.scatter(x, y, label='train')
ax.scatter(x_test, y_test, marker='*', label='test')
ax.set_ylim(-3, 3)

def fit_predict(degree):
    pf = PolynomialFeatures(degree)
    x_transformed = pf.fit_transform(x.reshape(-1, 1))
    xt_transformed = pf.transform(x_test.reshape(-1, 1))
    lr = LinearRegression()

    y_pred  = lr.fit(x_transformed.reshape(-1, 1), y).predict(x_transformed)
    yt_pred = lr.predict(xt_transformed.reshape(-1, 1))

    mse = mean_squared_error(y, y_pred)
    mse_test = mean_squared_error(yt_pred, y_test)

    return (y_pred, yt_pred, mse, mse_test)

# st.checkbox(label, value=False, 
# key=None, help=None, on_change=None, 
# args=None, kwargs=None, *, disabled=False, label_visibility="visible")
# if st.checkbox()

if st.checkbox('2', 2):
    fig = plt.subplots()

    # ax.scatter(x, y)
    # ax.scatter(x_test, y_test)

    poly = fit_predict(2)
    print(x.shape, poly[0].shape)
    print(x_test.shape, poly[1].shape)
    ax.plot(x, poly[0])
    ax.plot(x_test, poly[1])

    st.pyplot(fig)
    


    