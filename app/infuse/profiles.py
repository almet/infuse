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
from collections import defaultdict
import time

from progressbar import ProgressBar, Bar, Percentage
from scikits.learn.feature_extraction.text import Vectorizer, TfidfTransformer
from scikits.learn.decomposition import RandomizedPCA
from scikits.learn.cluster import (MeanShift, KMeans, AffinityPropagation, 
                                   SpectralClustering)
from scikits.learn.datasets import fetch_20newsgroups
from scikits.learn.metrics.pairwise import euclidean_distances

import numpy as np
import pylab as pl
import nltk

import db
from utils import mesure
from settings import OUTPUT_PATH

def find_profiles_text(algo=None, training_set=None, user=None):
    """Find different user profiles using the TF/IDF metric (Term Frequency / 
    Inverse Document Frequency).

    The stages of the pipeline are: 1. Vectorizer => 2. RandomizedPCA => 3. KMeans
    The use of the randomized PCA is useful here to reduce the dimensionality of the
    vectors space.

    As we lack some data, the dimentionality reduction is made using an already 
    existing dataset, the 20 newsgroup dataset.

    :parm algo: the algorithm to chose. Can be kmeans, meanshift or both (specified
                by "all")
    :param training_set: the training set to use for the word vectorisation.
                         The default setting is to use the 20 newsgroup dataset, 
                         it is possible to use the documents by specifying "docs"
    """
    # init some vars
    if not algo:
        algo = "all"
    if not training_set:
        training_set = "newsgroup"

    print "Computing clusters using the TF-IDF scores,"\
          " using %s algo and the %s training dataset" % (algo, training_set)

    # we first train the pca with all the dataset to have a most representative
    # model. Download the dataset and train the pca and the vector only if a 
    # pickled version is not available (i.e only during the first run).
    wide_dataset = docs = None

    vec_filename = os.path.join(OUTPUT_PATH, "pickle/vec-%s.pickle" % training_set)
    pca_filename = os.path.join(OUTPUT_PATH, "pickle/pca-%s.pickle" % training_set)
    pca2d_filename = os.path.join(OUTPUT_PATH, "pickle/pca2d-%s.pickle" % training_set)

    with mesure("  loading vectors"):
        if os.path.isfile(vec_filename):
            vec = _load_obj(vec_filename)
        else:
            docs = _load_docs(docs, training_set)
            vec = Vectorizer().fit(docs) # equivalent to CountVectorizer + TfIdf
            _save_obj(vec, vec_filename)

    with mesure("  loading PCA"):
        if os.path.isfile(pca_filename):
            pca = _load_obj(pca_filename)
        else:
            docs = _load_docs(docs, training_set)

            print "  reduce the dimentionality of the dataset to 100 components"
            # whiten=True ensure that the variance of each dim of the data in the 
            # transformed space is scaled to 1.0
            pca = RandomizedPCA(n_components=100, whiten=True).fit(vec.transform(docs))
            _save_obj(pca, pca_filename)

    # To visualize the data, we will project it on 2 dimensions. To do so, we 
    # will use a Principal Component Analysis (as we made in the first steps), 
    # but projecting on 2 dimensions.
    with mesure("  loading PCA 2D"):
        if os.path.isfile(pca2d_filename):
            pca_2d = _load_obj(pca2d_filename)
        else:
            docs = _load_docs(docs, training_set)
            print "  reduce the dimensionality of the dataset to 2 components"
            pca_2d = RandomizedPCA(n_components=2, whiten=True).fit(vec.transform(docs))
            _save_obj(pca_2d, pca2d_filename)

    # Now, go trough the whole resources for each users and try to find user 
    # profiles regarding TF-IDF
    # as the process can take some time, there is a progressbar to keep the user 
    # updated about the status of the operation
    for username in list(db.users.find().distinct('username')):
        if user and user != username:
            continue
        # get all the resources for this user
        urls = db.views.find({"user.username": username}).distinct("url")
        if not urls:
            continue # if we don't have any url for this user, go to the next one!

        resources = list(db.resources.find({'url': {'$in': urls }, 
            'blacklisted': False, 'processed': True}))
        if not resources:
            continue
        print "processing %s (%s docs)" % (username, len(resources))

        # get the docs content and names
        docs = [res['content'] for res in resources]
        urls = [res['url'] for res in resources]

        # fit the contents to the new set of features the PCA determined
        with mesure("  reduce dataset dimensions to 100"):
            docs_transformed = pca.transform(vec.transform(docs))

        # what we do have now is a matrix with 100 dimentions, which is not really 
        # useful for representation. Keeping this for later analysis is a good
        # thing so let's save this model for comparing profiles against resources
        # later
        # TODO pickle the kmeans into mongodb ?

        # project X onto 2D
        with mesure("  reduce dataset dimensions to 2"):
            docs_2d = pca_2d.transform(vec.transform(docs))

        # run the clustering algorithm
        if algo in ["kmeans", "all"]:
            with mesure("  kmeans(5)"):
                cluster = KMeans(k=5).fit(docs_transformed)

            # get_words_from_clusters(cluster, 10, docs, vec)
            # print "ngrams for km on %s" % username
            # get_n_bigrams_from_clusters(cluster, docs, 5)
            plot_2d(cluster, docs_2d, username, "kmeans", "Text-%s" % training_set)
            plot_pie(cluster, username, "kmeans", "Text-%s" % training_set)

        if algo in ["meanshift", "all"]:
            with mesure("  meanshift"):
                cluster = MeanShift().fit(docs_transformed) 
            # print "ngrams for ms on %s" % username
            # get_n_bigrams_from_clusters(cluster, docs, 3)
            plot_2d(cluster, docs_2d, username, "meanshift", "Text-%s" % training_set)
            plot_pie(cluster, username, "meanshift", "Text-%s" % training_set)

        if algo in ["affinity", "all"]:
            with mesure("  affinity propagation"):
                cluster = AffinityPropagation().fit(euclidean_distances(docs_transformed, docs_transformed))
            plot_pie(cluster, username, "affinity", "Text-%s" % training_set)

    # Once we've been trough all the users and got a number of different profiles
    # for each of them, there is a need to compare them and to find a similarity
    # measure between them.
    #
    # The goal is to compare each of the profiles against all the other ones.
    # To do so, there is a need to get a number of features that are representing
    # a profile.
    #
    # As we are based here on text features (we computed the TF-IDF score), 
    # carrying such information to describe profiles seems valuable. For each
    # profile, all the matching documents will be bundled together and the resulting
    # TF-IDF matrix will be projected onto the 20 newsgroup dataset; It will be then
    # possible to compare profiles against each others using pairwise metrics such 
    # the pearson correlation or the euclidean distance.


def find_profiles_context(algo=None, user=None):
    """Find profiles based on:
        * location of the views
        * time of the day of the views
        * time of the day
        * day of the week
    """
    if not algo:
        algo = "all"

    # get all users
    for username in db.users.distinct("username"):
        if user and user != username:
            continue

        urls = db.views.find({"user.username": username}).distinct("url")
        resources = []
        if not urls:
            continue
        print "processing %s (%s docs)" % (username, len(urls))
            
        t0 = time.time()
        progress = ProgressBar(
                    widgets=["  building the matrix for %s" % username, 
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
        print "matrix generation took %s" % (time.time() - t0)

        # project X on 2D
        # print "  project the dataset into 2d"
        # pca_2d = RandomizedPCA(n_components=2, whiten=True).fit(resources)
        # docs_2d = pca_2d.transform(resources)

        # run the clustering algorithm
        if algo in ["kmeans", "all"]:
            with mesure("  kmeans(5)"):
                cluster = KMeans(k=5).fit(resources)
            plot_2d(cluster, resources, username, "kmeans", "Context")
            plot_pie(cluster, username, "kmeans", "Context")

        if algo in ["meanshift", "all"]:
            with mesure("  meanshift"):
                cluster = MeanShift().fit(resources) 
            plot_2d(cluster, resources, username, "meanshift", "Context")
            plot_pie(cluster, username, "meanshift", "Context")

        if algo in ["affinity", "all"]:
            with mesure("  affinity propagation"):
                cluster = AffinityPropagation().fit(euclidean_distances(resources, resources))
            # plot_2d(cluster, resources, username, "affinity", "Context")
            plot_pie(cluster, username, "affinity", "Context")

def cluster_users(features=None):
    """Cluster the users, without using information about profiles.

    Different features can be used to do so, at least text features and context 
    features.
    """
    training_set="newsgroup"
    docs = None

    vec_filename = os.path.join(OUTPUT_PATH, "pickle/vec-%s.pickle" % training_set)
    pca_filename = os.path.join(OUTPUT_PATH, "pickle/pca-%s.pickle" % training_set)

    # get the training set, transform it to N dimensions
    with mesure("  loading vectors"):
        if os.path.isfile(vec_filename):
            vec = _load_obj(vec_filename)
        else:
            docs = _load_docs(docs, training_set)
            vec = Vectorizer().fit(docs) # equivalent to CountVectorizer + TfIdf
            _save_obj(vec, vec_filename)

    with mesure("  loading PCA"):
        if os.path.isfile(pca_filename):
            pca = _load_obj(pca_filename)
        else:
            docs = _load_docs(docs, training_set)

            print "  reduce the dimentionality of the dataset to 100 components"
            # whiten=True ensure that the variance of each dim of the data in the 
            # transformed space is scaled to 1.0
            pca = RandomizedPCA(n_components=100, whiten=True).fit(vec.transform(docs))
            _save_obj(pca, pca_filename)

    # for each user, get the contents related to him.
    users_content = []
    users_labels = []
    for username in list(db.users.find().distinct('username')):
        # get all the resources for this user
        urls = db.views.find({"user.username": username}).distinct("url")
        if not urls:
            continue # if we don't have any url for this user, go to the next one!

        resources = list(db.resources.find({'url': {'$in': urls }, 
            'blacklisted': False, 'processed': True}))
        if not resources:
            continue
        print "processing %s (%s docs)" % (username, len(resources))

        # get the docs content and names
        users_labels.append(username)
        users_content.append(" ".join([res['content'] for res in resources]))
    
    with mesure("  vectorise and reduce the dataset dimensions to 100"):
        transformed_content = pca.transform(vec.transform(users_content))

    # at the end, compute the similarity between users using different metrics
    # kmeans 3 clusters
    cluster = KMeans(3).fit(transformed_content)
    plot_pie(cluster, "all", "kmeans", "text")
    plot_2d(cluster, transformed_content, "all", "kmeans", "text")
    user_list = [[users_labels[idx] for idx, _ in enumerate(cluster.labels_ == cluster_id) if _] for cluster_id in np.unique(cluster.labels_)]

    # compute similarity scores
    from ipdb import set_trace; set_trace()


def plot_pie(cluster, username, algo, features):
    """Draw a pie shart with the percentages of documents in each cluster

    :param cluster: the clustering object
    :param username: the concerned username
    :param algo: the used algorithm (string)
    :param features: the name of the set of features used
    """
    fig = pl.figure(figsize=(6, 6))
    fig.clf()
    labels = range(max(cluster.labels_) + 1)
    fracs = np.bincount(cluster.labels_)
    pl.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True, 
            colors=[c for c in 'bgrcmyk'])
    pl.title("distribution for %s (%s, %s)" % (username, features, algo))
    fig.savefig(os.path.join(OUTPUT_PATH, 
        "figures/pie_%s_%s_%s.png" % (features.lower(), algo, username)))

def plot_2d(cluster, X, username, algo, features):
    """Plot the results on a 2D chart.
    
    :param cluster: the clustering object containting the labels and the centers
    :param X: the dataset used (should be in 2 dimensions)
    :param username: The username is used for both the title of the figure and 
                     the name of the file.
    :param algo: the algorithm that was used for clustering
    """
    labels = cluster.labels_
    cluster_centers = cluster.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    # create the figure
    fig = pl.figure(None)

    # clean the figure window
    pl.clf()
    ax = fig.add_subplot(111)

    colors = cycle(['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'bv', 'gv', 'rv', 'cv', 'mv', 'yv', 'kv'])
    for k, col in zip(range(n_clusters_), colors): # different color for each cluster

        # convert the matrix into a boolean matrix 
        # (to only get the point in this cluster)
        my_members = labels == k
        cluster_center = cluster_centers[k]

        # plot all the points belonging to this cluster
        # x and y axis are computed from a PCA and such can't be labeled
        ax.plot(X[my_members, 0], X[my_members, 1], col)

        # plot the centers
        ax.plot(cluster_center[0], cluster_center[1], 'o' if col[1] == '.' else col[1], 
                markerfacecolor=col[0], markeredgecolor='k', markersize=14)

        ax.set_title('%s: %s clusters using %s' % (algo, n_clusters_, features))
    fig.savefig(os.path.join(OUTPUT_PATH, 
        "figures/2d_%s_%s_%s.png" % (features.lower(), algo, username)))


def get_n_bigrams_from_clusters(cluster, docs, N):
    """Return the N most used bigrams for each of the clusters in the given
    clustering object.

    :param cluster: the clustering object
    :param docs: the documents to filter on
    :param N: the number of bigrams to return
    """
    bigrams = {}
    for idx in np.unique(cluster.labels_):
        tokens =  [token for item 
                   in [doc for i, doc in enumerate(docs) if (cluster.labels_ == idx)[i]] 
                   for token in item.lower().split()]
        finder = nltk.BigramCollocationFinder.from_words(tokens)

        # Only return the bigrams that are appearing at least 3 times in the corpus
        finder.apply_freq_filter(3) 

        bigrams[idx] = finder.nbest(nltk.metrics.BigramAssocMeasures.jaccard, N)
    return bigrams


def get_words_from_clusters(cluster,  n, docs, vec):
    """Return the N most used words per cluster

    TODO: Find a better and quicker implementation!
    """
    terms = dict([(id, term) for term, id in vec.vocabulary.items()])
    
    clusters = {}
    for idx in np.unique(cluster.labels_):
        cluster_docs = [doc for i, doc in enumerate(docs) 
                if (cluster.labels_ == idx)[i]]
        vectorized_docs = vec.transform(cluster_docs)
        clusters[idx] = terms[vectorized_docs.sum(axis=0).argmax()]
        print clusters[idx]
    return clusters

# below are some utility fonctions to save/load python objects to/from disc
def _load_docs(docs, training_set):
    if not docs:
        if training_set == "newsgroup":
            print "  extract the 20 newsgroup dataset"
            wide_dataset = fetch_20newsgroups()
            docs = [open(f).read() for f in wide_dataset.filenames]
        elif training_set == "docs":
            docs = [res['content'] for res in 
                    db.resources.find({'blacklisted': False, 
                                       'processed': True})]
    return docs


def _load_obj(filename):
    with open(filename) as f:
        print "  unpickle %s" % filename
        return pickle.load(f)

def _save_obj(obj, filename):
    with open(filename, "w+") as f:
        print "  save %s" % filename
        pickle.dump(obj, f)

def main(action, options):
    if not action:
        action = "text"
    if not options:
        options = []

    if action == "text":
        find_profiles_text(*options)
    elif action == "context":
        find_profiles_context(*options)
    elif action == "cluster":
        cluster_users(*options)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        action = sys.argv[1]
    else:
        action = "text"

    if len(sys.argv) >= 3:
        options = sys.argv[2:]
    else:
        options = None

    main(action, options)
