import pickle
import os

from scikits.learn.feature_extraction.text import Vectorizer
from scikits.learn.cluster import KMeans, MeanShift, estimate_bandwidth
from scikits.learn.decomposition import RandomizedPCA
from scikits.learn.datasets import fetch_20newsgroups
import numpy as np

import db
from recommender import get_views_features
from drawing import draw_2d, draw_cluster_2d, compare_pies
from settings import OUTPUT_PATH

def get_text_dataset(username, use_20newsgroup=False):
    """Return the non empty resources visited by this user"""
    # get all the views by this user
    views = list(db.views.find({"user.username": username}).distinct("url"))
    dataset = [r['content'] for r in db.resources.find({"url": {"$in": views}}) 
                if r['content']]

    # get the features
    docs = "newsgroup" if use_20newsgroup else "docs"
    
    # load pickled objects as it is really long to compute otherwise
    with open(os.path.join(OUTPUT_PATH, "pickle",
        "vec%s.pickle" % docs)) as f:
        vec = pickle.load(f)

    with open(os.path.join(OUTPUT_PATH, "pickle", 
        "pca100-%s.pickle" % docs)) as f:
        pca = pickle.load(f)

    return pca.transform(vec.transform(dataset))

def get_context_dataset(username, advanced=False):
    views = list(db.views.find({"user.username": username}))
    return np.array(get_views_features(views, advanced))

def main(username):
    """Compare different clustering algorithms / configurations for the two
    existing datasets (context and text)
    """
    def _draw(dataset, filename, title):
        pca = RandomizedPCA(2)
        pca.fit(dataset)
        X = pca.transform(dataset)
        draw_2d(X, filename, title)

    def _kmeans(*ks):
        """utility function to return instances of kmeans with a predefined
        number of clusters.

        the passed list is the K value for the clusters to return
        
        """
        if not ks:
            ks = [5, 10, 20, 50]

        instances = []
        for k in ks:
            instances.append(KMeans(k))
        return instances

    def _meanshift(dataset, *bandwidths):
        if not bandwidths:
            bandwidths = [0.1, 0.3, 0.5, 0.7, 0.9]
        instances = []
        for bw in bandwidths:
            instances.append(MeanShift(estimate_bandwidth(dataset, bw)))
        return instances

    def _compare_clusters(**datasets):
        for name, dataset in datasets.items():
            pca = RandomizedPCA(2)
            pca.fit(dataset)
            X = pca.transform(dataset)
            instances = _kmeans()
            for instance in instances:
                instance.fit(dataset)
                # reduce to 2d for visualisation
                draw_cluster_2d(instance, X, 
                        filename="%s-kmeans-%s.png" % (name, instance.k))
            ms_instances = _meanshift(dataset)
            for instance in ms_instances:
                instance.fit(dataset)
            compare_pies(
                    [_get_distribution(i) for i in instances] + 
                        [_get_distribution(i) for i in ms_instances],
                    ["KMeans(%s)" % i.k for i in instances] + 
                        ["MeanShift(%s)" % round(i.bandwidth) for i in ms_instances],
                    filename="%s-pie.png" % name)

    def _get_distribution(cluster):
        return [len([i for i in cluster.labels_ == label if i]) 
                for label in np.unique(cluster.labels_)]


    # start by getting the three datasets
    print "load text dataset"
    text = get_text_dataset(username)
    print "load text dataset (vectorized with newsgroup)"
    text_ng = get_text_dataset(username, use_20newsgroup=True)
    print "load context dataset"
    context = get_context_dataset(username)

    # draw them
    _draw(text, "text_2d.png", "2D text features")
    _draw(text_ng, "text_ng_2d.png", "2D NG-text features")
    _draw(context, "context_2d.png", "2D context features")

    # And try different clustering techniques on them
    _compare_clusters(text=text, text_ng=text_ng, context=context)
    
if __name__ == "__main__":
    main("alexis")
