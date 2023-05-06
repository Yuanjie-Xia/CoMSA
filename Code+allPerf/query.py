from collections import Counter
from typing import Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax
from modAL.models.base import BaseCommittee

from sklearn.preprocessing import minmax_scale

def max_std_sampling(regressor: BaseEstimator, X: modALinput,
                     n_instances: int = 1,  random_tie_break=False,
                     **predict_kwargs) -> np.ndarray:
    """
    Regressor standard deviation sampling strategy.

    Args:
        regressor: The regressor for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **predict_kwargs: Keyword arguments to be passed to :meth:`predict` of the CommiteeRegressor.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    _, std = regressor.predict(X, return_std=True, **predict_kwargs)
    std = std.reshape(X.shape[0], )

    if not random_tie_break:
        return multi_argmax(std, n_instances=n_instances), std

    return shuffled_argmax(std, n_instances=n_instances), std

def flexible_weight(weight, add_idx, query_instance=1):
    if query_instance == 1:
        index = [i[0] for i in sorted(enumerate(weight), key=lambda x:x[1])]
        for i in range(1, len(index)):
            # print("added idx: " + str(add_idx))
            # print("item repeated time: " + str(add_idx.count(index[-i])))
            if add_idx.count(index[-i]) < 1:
                j = index[-i]
                break
            else:
                continue
    else:
        pass
    return [j], query_instance


def main():
    config = [1, 2, 3]
    weight = [5, 8, 1]
    flexible_weight(config, weight)


if __name__ == "__main__":
    main()
