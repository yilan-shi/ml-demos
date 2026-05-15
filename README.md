# ML from Scratch ⚙️

Interactive demo of machine learning algorithms built from scratch (using Numpy). Has sliders for learning rate and training steps, with some stats/probability concepts.   
Work is derived from own code from a prior CS hw assignment at Stanford (CS109, Professor Chris Piech, Spring 2025 quarter). 

## Five tabs showing work from five (anonymized) datasets:

| Tab | Algorithm | Dataset | What it shows |
|-----|-----------|---------|---------------|
| Logistic Regression | Gradient ascent, NumPy only | 2-feature toy | Decision boundary, convergence curve |
| Ancestry Classifier | Logistic regression on SNPs | 20-column binary genomic data | Feature by weight magnitude |
| Heart Disease | Learning rate grid search | 22 medical features, 79 patients | Best η from 8 candidates |
| Calibration | Bucket analysis | Ancestry calibration set | Empirical vs predicted probability |
| Caltrain Ridership | Linear regression + MLE | 1,641 Caltrain observations | P(model is off by > 20 passengers) |

---

## Big overarching idea of what I did: 

Here I implemented logistic regression from scratch — no sklearn for the model itself (we were forbidden from using packages to enforce/encourage learning from first principles). 

The training loop is a gradient ascent on the log-likelihood:

```python
def train_logreg(X, y, lr=0.0001, steps=1000):
    X_b = np.hstack([X, np.ones((X.shape[0], 1))])   # add bias column
    w   = np.zeros(X_b.shape[1])
    for _ in range(steps):
        h = sigmoid(X_b @ w)                          # forward pass
        w += lr * (X_b.T @ (y - h))                  # gradient ascent step
    return w
```

The Caltrain regression uses sklearn's `LinearRegression`, derives the
Maximum Liklihood Estimate (MLE) for residual variance, and then computes the probability of large errors using a Normal CDF.

---

## Details:
Tab 1: logistic regression from scratch using a small dataset (2 features only) from simple-train and train-test.csv. This is a gradient ascent on log-liklihood, moving weights in the direction that increases liklihood of observed labels. 

Data treatment: I loaded the data with pandas, separated features in the Label column, and converted labels to floats. 

Logistic Reg main function: took feature matrix X, labels y, a learning rate and # of steps, and appends a column of 1's to X so the bias term get learned alongisde feature wts. This allowed me to write Xw instead of Xw + b. Then I initialized all weights to zero, run gradient ascent for certain # of steps, and calculated probabilities with sigmoid(X @ w). So gradient = X.T@(y-h) with y-h being the error, and updated with weights += learning rate *gradient calculated. 

---

## Stack

Python · NumPy · pandas · matplotlib · Streamlit · scikit-learn · scipy

---

The source code is in the .py file and encrypted. Sorry, it is for academic integrity purposes. Can share upon request if needed for evaluation of jobs, internships, and fellowships. Please contact yilan.shi@gmail.com. Thanks! 

The original code is encrypted to respect academic integrity policies, so only a front-end demo is shown here. For consideration of fellowships, jobs, internships...etc, please contact yilan.shi@gmail.com. Thanks!
