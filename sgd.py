#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

import pickle
import glob
import random
import numpy as np
import os
from tqdm import tqdm

def load_saved_params(path="./saved_model"):
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    for f in glob.glob(os.path.join(path, "saved_params_*.npy")):
        iter = int(os.path.splitext(os.path.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        params_file = "saved_params_%d.npy" % st
        state_file = "saved_state_%d.pickle" % st
        params = np.load(os.path.join(path, params_file))
        with open(os.path.join(path, state_file), "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params, path="./saved_model"):
    if not os.path.exists(path):
        os.mkdir(path)
    params_file = os.path.join(path, "saved_params_%d.npy" % iter)
    np.save(params_file, params)
    with open(os.path.join(path, "saved_state_%d.pickle" % iter), "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    This function implements the stochastic gradient descent method.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            print('loaded saved model params...')
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    # start model training
    print('starting model training with the following parameters:')
    print(' - learning rate:', step)
    print(' - iterations:', iterations)
    print(' - start_iter:', start_iter)
    print(' - ANNEAL_EVERY:', ANNEAL_EVERY)
    print(' - PRINT_EVERY:', PRINT_EVERY)
    print('=' * 40)

    pbar = tqdm(range(start_iter + 1, iterations + 1), desc='Training => Loss: %.6f' % np.Inf)
    for iter in pbar:
        loss, grad = f(x)
        x = x - (step * grad)

        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            pbar.set_description("Training => loss: %.6f" % exploss)

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    assert abs(t3) <= 1e-6

    print("-" * 40)
    print("ALL TESTS PASSED")
    print("-" * 40)


if __name__ == "__main__":
    sanity_check()