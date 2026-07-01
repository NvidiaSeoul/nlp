"""results/ 시각화 재현 스크립트 — scikit-learn + matplotlib
실행: python make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

R = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(R, exist_ok=True)
iris = load_iris()
X, y = iris.data[:, 2:4], iris.target
gnb = GaussianNB().fit(X, y)
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 400),
                     np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 400))
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, Z, alpha=.25, cmap="coolwarm")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=35)
ax.set(xlabel="Petal length (cm)", ylabel="Petal width (cm)",
       title="Gaussian Naive Bayes — Iris Decision Regions")
fig.tight_layout(); fig.savefig(f"{R}/gaussian_nb_iris.png", dpi=120); plt.close(fig)
print("saved figures to", R)
