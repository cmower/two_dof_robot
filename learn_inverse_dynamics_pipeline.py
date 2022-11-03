import numpy as np
import matplotlib.pyplot as plt

from learn_inverse_dynamics_data_collection import main as main_data_collection
from learn_inverse_dynamics_training import main as main_training
from learn_inverse_dynamics_test import main as main_test


def main():

    Ntrials_test = np.arange(100, 1001, 100)
    err1_results = []
    err2_results = []

    for Ntrials in Ntrials_test:
        data_filename = main_data_collection(animate=False, Ntrials=Ntrials)
        nn_filename = main_training(data_filename)
        results = main_test(nn_filename, verbose_output=False, animate=False, plot_cmp=False)

        err1 = np.linalg.norm(results['ptauext1'] - results['tauext1'])
        err2 = np.linalg.norm(results['ptauext2'] - results['tauext2'])

        err1_results.append(err1)
        err2_results.append(err2)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(Ntrials_test, err1_results, '-o', label='err1')
    ax.plot(Ntrials_test, err2_results, '-o', label='err2')

    plt.show()


if __name__ == '__main__':
    main()
