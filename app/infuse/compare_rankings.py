"""
This module tests the accuracy of the ranking inferance.
It mainly tests two things:
    * The mean squarred error (MSE)
    * The mean absolute error (MAE)

The above metrics are run on an annotated set of data. The dataset is divided into
two parts: the training set and the real set. We then check that what is infered
is compatible with the real values.
"""
from collections import defaultdict
from math import sqrt, pow

from recommender import get_rankings, get_ranked_dataset, get_views_from_resources
import db

def root_mean_squared_error(estimated, actual):
    return float(sqrt(sum([pow(estimated[resource] - label, 2)
                     for resource, label in actual.items()])) / len(actual))

def mean_absolute_error(estimated, actual):
    return float(sum([abs(estimated[resource] - label)
               for resource, label in actual.items()])) / len(actual)

def test_classifier():
    username = "alexis" # the only one with annotated data

    # get the ranked resources
    temp = get_ranked_dataset(username)

    # split in two equal parts
    training_set, real_set = temp[:len(temp)/2], temp[len(temp)/2:]

    rankings = get_rankings([username,], lambda x: training_set, lambda x: real_set)
    # now compare the returned rankings with the real ones
    with_labels = {}
    for resource in real_set:
        with_labels[resource['url']] = \
                int(db.views.find({ 'url': resource['url'], 
                    'user.username': username }).distinct('feedback')[0])
   
    def _confusion_matrix(values):
        matrix = defaultdict(int)
        for url, value in values.items():
            matrix[round(value)] += 1
        return matrix

    #actual = _confusion_matrix(with_labels)
    #jpredicted = _confusion_matrix(dict([(k, v) for k,v in rankings[username].items()
    #                                    if k in with_labels.keys()]))

    rmse = root_mean_squared_error(rankings['alexis'], with_labels)
    mae = mean_absolute_error(rankings['alexis'], with_labels)
    return mae, rmse

def test_estimator():
    username = 'alexis'
    dataset = get_ranked_dataset(username)
    rankings = get_rankings([username,], lambda x: [], lambda x:dataset)

    with_labels = {}
    for resource in dataset:
        with_labels[resource['url']] = \
                int(db.views.find({ 'url': resource['url'], 
                    'user.username': username }).distinct('feedback')[0])

    # once we get the estimations, compare them to the actual values
    rmse = root_mean_squared_error(rankings['alexis'], with_labels)
    mae = mean_absolute_error(rankings['alexis'], with_labels)
    return mae, rmse

if __name__ == "__main__":
    estimator = test_estimator()
    classifier = test_classifier()
    from ipdb import set_trace; set_trace()
