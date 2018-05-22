import numpy as np
import random
import torch
from net import SineModel
from DataLoader import SineWaveTask
from tools import sine_fit1, plot_sine_test, plot_sine_learning, maml_sine, reptile_sine
import matplotlib.pyplot as plt

TRAIN_SIZE = 10000
TEST_SIZE = 1000

SINE_TRAIN = [SineWaveTask() for _ in range(TRAIN_SIZE)]
SINE_TEST = [SineWaveTask() for _ in range(TEST_SIZE)]

SINE_TRANSFER = SineModel()


def fit_transfer(epochs=1):
    optim = torch.optim.Adam(SINE_TRANSFER.params())

    for _ in range(epochs):
        for t in random.sample(SINE_TRAIN, len(SINE_TRAIN)):
            sine_fit1(SINE_TRANSFER, t, optim)


def main():

    # Mean And Random Version #
    ONE_SIDED_EXAMPLE = None
    while ONE_SIDED_EXAMPLE is None:
        cur = SineWaveTask()
        x, _ = cur.training_set()
        x = x.numpy()
        if np.max(x) < 0 or np.min(x) > 0:
            ONE_SIDED_EXAMPLE = cur

    fit_transfer()

    plot_sine_test(SINE_TRANSFER, SINE_TEST[0], fits=[0, 1, 10], lr=0.02)

    plot_sine_learning(
        [('Transfer', SINE_TRANSFER), ('Random', SineModel())],
        list(range(100)),
        marker='',
        linestyle='-', SINE_TEST=SINE_TEST)

    # MaML #
    SINE_MAML = [SineModel() for _ in range(5)]

    for m in SINE_MAML:
        maml_sine(m, 4, SINE_TRAIN=SINE_TRAIN)

    plot_sine_test(SINE_MAML[0], SINE_TEST[0], fits=[0, 1, 10], lr=0.01)
    plt.show()

    plot_sine_learning(
        [('Transfer', SINE_TRANSFER), ('MAML', SINE_MAML[0]), ('Random', SineModel())],
        list(range(10)),
        SINE_TEST=SINE_TEST
    )
    plt.show()

    plot_sine_test(SINE_MAML[0], ONE_SIDED_EXAMPLE, fits=[0, 1, 10], lr=0.01)
    plt.show()


    # First Order #
    SINE_MAML_FIRST_ORDER = [SineModel() for _ in range(5)]

    for m in SINE_MAML_FIRST_ORDER:
        maml_sine(m, 4, first_order=True, SINE_TRAIN=SINE_TRAIN)

    plot_sine_test(SINE_MAML_FIRST_ORDER[0], SINE_TEST[0], fits=[0, 1, 10], lr=0.01)
    plt.show()

    plot_sine_learning(
        [('MAML', SINE_MAML), ('MAML First Order', SINE_MAML_FIRST_ORDER)],
        list(range(10)),
        SINE_TEST=SINE_TEST
    )
    plt.show()

    plot_sine_test(SINE_MAML_FIRST_ORDER[0], ONE_SIDED_EXAMPLE, fits=[0, 1, 10], lr=0.01)
    plt.show()

    # Reptile #
    SINE_REPTILE = [SineModel() for _ in range(5)]

    for m in SINE_REPTILE:
        reptile_sine(m, 4, k=3, batch_size=1, SINE_TRAIN=SINE_TRAIN)

    plot_sine_test(SINE_REPTILE[0], SINE_TEST[0], fits=[0, 1, 10], lr=0.01)
    plt.show()

    plot_sine_learning(
        [('MAML', SINE_MAML), ('MAML First Order', SINE_MAML_FIRST_ORDER), ('Reptile', SINE_REPTILE)],
        list(range(32)),
        SINE_TEST=SINE_TEST
    )
    plt.show()

    plot_sine_test(SINE_REPTILE[0], ONE_SIDED_EXAMPLE, fits=[0, 1, 10], lr=0.01)
    plt.show()


if __name__ == "__main__":
    main()
