"""Cluster user resource views to find profiles.

For each user, at some point, we will try to determine a number of clusters 
(profiles).

This module propose different ways to find profiles:

    * using TF/IDF on web pages contents (1)
    * using bag of words in the subject of the page (2)

And using all those criterias at once (3):

    * using the location from where the views of the resources have been made
    * Using the time of the day and the day of week of the view
    * Using the total duration of the resource for this user

For each one, a different clustering algorithm can beused: KMeans, MeanShift, 
Affinity propagation, Spectral clustering. All of them are unsupervised learning
(e.g we don't have a set of groups we already know and want to learn about)
"""

from itertools import cycle
import sys
import os
import pickle

from progressbar import ProgressBar, Bar, Percentage
from scikits.learn.feature_extraction.text import Vectorizer, TfidfTransformer
from scikits.learn.decomposition import RandomizedPCA
from scikits.learn.cluster import estimate_bandwidth, MeanShift, KMeans
from scikits.learn.datasets import fetch_20newsgroups
import numpy as np
import pylab as pl

import db
from settings import OUTPUT_PATH

def find_profiles_tfidf(algo=None):
    """Find different user profiles using the TF/IDF metric (Term Frequency / 
    Inverse Document Frequency).

    The stages of the pipeline are: 1. Vectorizer => 2. RandomizedPCA => 3. KMeans
    The use of the randomized PCA is useful here to reduce the dimensionality of the
    vectors space.

    As we lack some data, the dimentionality reduction is made using an already 
    existing dataset, the 20 newsgroup dataset.

    :parm algo: the algorithm to chose. Can be kmeans, meanshift
    """

    def _load_docs(docs):
        if not docs:
            print "extract the 20 newsgroup dataset"
            wide_dataset = fetch_20newsgroups()
            docs = [open(f).read() for f in wide_dataset.filenames]
        return docs

    def _load_obj(filename):
        with open(filename) as f:
            print "unpickle %s" % filename
            return pickle.load(f)

    def _save_obj(obj, filename):
        with open(filename, "w+") as f:
            print "save %s" % filename
            pickle.dump(obj, f)

    # init
    if not algo:
        algo = "all"

    # we first train the pca with all the dataset to have a most representative
    # model. Download the dataset and train the pca and the vector only if a 
    # pickled version is not available (i.e only during the first run).
    wide_dataset = docs = None

    vec_filename = os.path.join(OUTPUT_PATH, "pickle/vec.pickle")
    pca_filename = os.path.join(OUTPUT_PATH, "pickle/pca.pickle")
    pca2d_filename = os.path.join(OUTPUT_PATH, "pickle/pca2d.pickle")

    if os.path.isfile(vec_filename):
        vec = _load_obj(vec_filename)
    else:
        docs = load_docs(docs)
        vec = Vectorizer().fit(docs) # equivalent to CountVectorizer + TfIdf
        _save_obj(vec, vec_filename)

    if os.path.isfile(pca_filename):
        pca = _load_obj(pca_filename)
    else:
        docs = load_docs(docs)

        print "reduce the dimentionality of the dataset to 100 components"
        # whiten=True ensure that the variance of each dim of the data in the 
        # transformed space is scaled to 1.0
        pca = RandomizedPCA(n_components=100, whiten=True).fit(vec.transform(docs))
        _save_obj(pca, pca_filename)

    # To visualize the data, we will project it on 2 dimensions. To do so, we 
    # will use a Principal Component Analysis (as we made in the first steps), 
    # but projecting on 2 dimensions.
    if os.path.isfile(pca2d_filename):
        pca_2d = _load_obj(pca2d_filename)
    else:
        docs = _load_docs(docs)
        print "reduce the dimensionality of the dataset to 2 components"
        pca_2d = RandomizedPCA(n_components=2, whiten=True).fit(vec.transform(docs))
        _save_obj(pca_2d, pca2d_filename)

    # Now, go trough the whole resources for each users and try to find user 
    # profiles regarding TF-IDF
    # as the process can take some time, there is a progressbar to keep the user 
    # updated about the status of the operation
    progress = ProgressBar(widgets=["cluster user profiles", Percentage(), Bar()])
    for username in progress(list(db.users.find().distinct('username'))):
        # 2. get all the resources for this user
        urls = db.views.find({"user.username": username}).distinct("url")
        if not urls:
            continue # if we don't have any url for this user, go to the next one!
        resources = list(db.resources.find({'url': {'$in': urls }, 
            'blacklisted': False, 'processed': True}))
        if not resources:
            continue

        # get the docs content and names
        docs = [res['content'] for res in resources]
        urls = [res['url'] for res in resources]

        # fit the contents to the new set of features the PCA determined
        docs_transformed = pca.transform(vec.transform(docs))

        # what we do have now is a matrix with 100 dimentions, which is not really 
        # useful for representation. Keeping this for later analysis is a good
        # thing so let's save this model for comparing profiles against resources
        # later
        # TODO pickle the kmeans into mongodb ?

        # project X onto 2D
        docs_2d = pca_2d.transform(vec.transform(docs))

        # run the clustering algorithm
        if algo == "kmeans" or "all":
            cluster = KMeans(k=5).fit(docs_transformed)
            plot_results(cluster, docs_2d, username, "kmeans", "TF-IDF")

        if algo == "meanshift" or "all":
            cluster = MeanShift().fit(docs_transformed) 
            plot_results(cluster, docs_2d, username, "meanshift", "TF-IDF")


def find_profiles_others(algo=None):
    """Find profiles based on:
        * location of the views
        * time of the day of the views
        * time of the day
        * day of the week
    """
    if not algo:
        algo = "all"

    # get all users
    for username in list(db.users.find()).distinct("username"):
        urls = db.views.find({"user.username": username}).distinct("url")
        resources = []
        if not urls:
            continue
            
        progress = ProgressBar(
                    widgets=["constructing the matrix for %s" % username, 
                    Percentage(), Bar()])
        for url in progress(urls):
            # get the views related to this user and this url
            views = db.views.find({"user.username": username, "url": url})

            views = list(views)
            indicators = ['average', 'mean', 'median', 'var', 'std']

            row = [len(views), sum([int(v['duration']) for v in views])]
            # TODO add location

            daytimes = []
            weekdays = []
            for view in views:
                daytimes.append(view['daytime'])
                weekdays.append(view['weekday'])

            for indicator in indicators:
                row.append(getattr(np, indicator)(daytimes))
                row.append(getattr(np, indicator)(weekdays))
            resources.append(row)

        resources = np.array(resources)

        # project X on 2D
        pca_2d = RandomizedPCA(n_components=2, whiten=True).fit(resources)
        docs_2d = pca_2d.transform(resources)

        # run the clustering algorithm
        if algo == "kmeans" or "all":
            cluster = KMeans(k=5).fit(resources)
            plot_results(cluster, docs_2d, username, "kmeans", "views")

        if algo == "meanshift" or "all":
            cluster = MeanShift().fit(resources) 
            plot_results(cluster, docs_2d, username, "meanshift", "views")


def plot_results(ms, X, filename, algo, features):
    """Plot the results on a 2D chart.
    
    :param cluster: the clustering object containting the labels and the centers
    :param X: the dataset used (should be in 2 dimensions)
    :param filename: The filename is used for both the title of the figure and 
                     the name of the file.
    :param algo: the algorithm that was used for clustering
    """
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    # create the figure
    fig = pl.figure(None)

    # clean the figure window
    pl.clf()
    ax = fig.add_subplot(111)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors): # different color for each cluster

        # convert the matrix into a boolean matrix 
        # (to only get the point in this cluster)
        my_members = labels == k
        cluster_center = cluster_centers[k]

        # plot all the points belonging to this cluster
        # x and y axis are computed from a PCA and such can't be labeled
        ax.plot(X[my_members, 0], X[my_members, 1], col + '.')

        # plot the centers
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                        markeredgecolor='k', markersize=14)

    ax.set_title('2D PCA projection of %s %s corpus (with %s clusters, using %s)' % 
            (filename, features, n_clusters_, algo))
    fig.savefig(os.path.join(OUTPUT_PATH, 
        "figures/%s_%s_%s.png" % (features, algo, filename)))



def main(action="tfidf", *options):
    if action == "tfidf":
        find_profiles_tfidf(*options)
    elif action == "views":
        find_profiles_others()

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        action = sys.argv[1]
    else:
        action = "tfidf"

    if len(sys.argv) >= 3:
        algo = sys.argv[2]
    else:
        algo = None

    main(action, algo)
