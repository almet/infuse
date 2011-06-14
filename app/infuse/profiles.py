"""Cluster user interests to find profiles

Each profile will have a set of words and will be connected to an user.
Profiles will be stored in the database and resources will be attached to
profiles, not users.

To cluster users, we use a number of steps:

    * get the list of all users
    * for each user, get the list of resources he visited
    * for each resource, compute the TF/IDF score, for the whole corpus
    * Define K profiles for each user, using the different algorithms.
      Idealy, each algorithm could output its results as a graph. Those
      will be compared to find the best one to use.
"""

from itertools import cycle
from progressbar import ProgressBar
from scikits.learn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scikits.learn.cluster import estimate_bandwidth, MeanShift
import numpy as np
import pylab as pl


import db

def find_profiles():
    """Cluster users into profiles"""

    progress = ProgressBar()
    # 1. get all the users
    for user in progress(list(db.users.find())):
        # 2. get all the resources for this user
        # resources are not directly related to users, to it is
        # needed to go trough the views
        urls = db.views.find({"user.username": user['username']}).distinct("url")
        if not urls:
            continue
        resources = list(db.resources.find({'url': {'$in': urls }, 'blacklisted': False, 'processed': True}))

        # 3. compute the TF/IDF for the whole corpus
        count_vect = CountVectorizer()

        # get the docs content
        docs = [res['content'] for res in resources]

        # fit the contents to a vector of words
        count_vect.fit(docs)

        # perform the actual feature extraction
        x_counts = count_vect.transform(docs)

        # get Term frequencies
        tf_transformer = TfidfTransformer(use_idf=False).fit(x_counts)
        X_tf = tf_transformer.transform(x_counts)

        # get inverse document frequencies
        tfidf_transformer = TfidfTransformer().fit(x_counts)
        X_tfidf = tfidf_transformer.transform(x_counts)

        # 4. cluster the findings (with default settings for now)
        X_tfidf = X_tfidf.toarray()
        bandwidth = estimate_bandwidth(X_tfidf)
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(X_tfidf)

        # 5. Plot the results before creating user profiles
        plot_results(ms, X_tfidf, user['username'])

 
def plot_results(ms, X, filename):
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        pl.figure(1)
        pl.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            pl.plot(X[my_members,0], X[my_members,1], col+'.')
            pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                            markeredgecolor='k', markersize=14)
        pl.title('Estimated number of clusters: %d' % n_clusters_)
        pl.savefig("%s.png" % filename)



def main():
    find_profiles()

if __name__ == '__main__':
    main()
