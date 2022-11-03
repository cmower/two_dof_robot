import os
import sys
import time
import pickle
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def main(filename):

    df = pd.read_csv(filename)
    X = df[['theta1', 'theta2', 'dtheta1', 'dtheta2', 'ddtheta1', 'ddtheta2']].values
    y = df[['tau1', 'tau2']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    regr = MLPRegressor(
        hidden_layer_sizes=(100,),
        random_state=1,
        max_iter=500,
        verbose=True,
    ).fit(X_train, y_train)

    print("Train score:", regr.score(X_train, y_train))
    print("Test score:", regr.score(X_test, y_test))

    nndir = os.path.join('.', 'nn')
    if not os.path.exists(nndir):
        os.mkdir(nndir)

    stamp = time.time_ns()
    filename = os.path.joint(nndir, f'learn_id_nn_{stamp}.nn')
    with open(filename, 'wb') as f:
        pickle.dump(regr, f)
    print("Saved", filename)

    return filename

if __name__ == '__main__':
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Filename must be given.")
        sys.exit(0)    
    main(filename)
