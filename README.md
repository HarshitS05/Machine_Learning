# 🧠 Machine Learning — From Scratch, No Shortcuts

> Building ML algorithms from first principles, one line of NumPy at a time — because copy-pasting `sklearn.fit()` teaches you nothing.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-from--scratch-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-data%20wrangling-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 The Story

This repo is my hands-on journey through classical machine learning — not a course I audited, but a set of algorithms I broke, debugged, and rebuilt until they actually made sense. No black boxes. Every model here started as math on paper before it became code.

I'm an engineering student at **MIT Manipal**, part of **Project Manas** (my college's AI & robotics team), and building AI products for the Indian market on the side. This repo is where the fundamentals get forged — regression, trees, ensembles, clustering — the stuff that makes everything downstream (RAG pipelines, CNNs, agents) actually click.

---

## 🚀 What's Inside

| Project | Concept | Dataset |
|---|---|---|
| `linear_regression.py` | Linear Regression from scratch | — |
| `logistic_regression.py` | Logistic Regression — binary classification | — |
| `spam_mail_classifier.py` | Logistic Regression applied to spam detection | `spamhamdata.csv` |
| `decision_trees.py` | Decision Trees — entropy, Gini index, information gain, recursive splitting | `crime_train.csv` |
| `ensemble.py` | Ensemble Methods — bagging/boosting logic built manually | `yeast.csv` |
| `K.py` | K-Means Clustering — unsupervised, from-scratch centroid updates | `cluster_data.csv` |

Every `.py` file is self-contained: load the paired dataset, run it, watch the model learn.

---

## 🛠️ Tech Stack

- **Python** — the only dependency that matters
- **NumPy** — the math, done manually (gradients, distance metrics, entropy calculations)
- **Pandas** — data loading and preprocessing
- No `scikit-learn` shortcuts for the core algorithms — that defeats the point

---

## ⚡ Quick Start

```bash
git clone https://github.com/HarshitS05/Machine_Learning.git
cd Machine_Learning
pip install numpy pandas

# run any project
python linear_regression.py
python decision_trees.py
python K.py
```

Each script reads its paired `.csv` from the same directory — no config needed, just run and go.

---

## 🎯 Why "From Scratch"?

Because understanding *why* a decision tree picks a split, or *why* gradient descent converges (or doesn't), is worth infinitely more than an `import` statement. Every project here forced me to actually understand:

- How gradients flow in linear/logistic regression
- How entropy and information gain decide a tree's structure
- How ensembling reduces variance without a magic wrapper function
- How centroids converge in unsupervised clustering

This repo is a running log of that understanding — built, broken, and rebuilt until it worked.

---

## 🔭 What's Next

More algorithms are on the way as the learning continues — neural networks from scratch, and beyond. Watch this space.

---

## 🤝 Connect

Building AI products, exploring ML fundamentals, and occasionally breaking backprop on purpose to understand it better.

**Harshit** · MIT Manipal · Project Manas
