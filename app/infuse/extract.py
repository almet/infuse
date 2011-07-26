from itertools import cycle
from contextlib import contextmanager
import sys
import os
import pickle
from collections import defaultdict
from time import time

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

"""
This module defines classes able to extract information from the stored documents.

It has been created mainly to provide structure and reusable bits of code, in
order to stick to the DRY (Don't repeat yourself) principle.

This module introduces a super class "Clusterer", which aims to be a base class
for clusterer classes.
"""

class Clusterer(object):

    _features = "all"
    _clusters = { "kmeans": KMeans, "meanshift": MeanShift }
    _draw_pie = True
    _draw_2d = True
    
    def __init__(self, particular_user=None):
        self._particular_user = particular_user
        self._level = 0

    def object_filename(self, obj_name):
        return os.path.join(OUTPUT_PATH, "pickle/%s.pickle" % obj_name)

    def indent(self):
        self._level = self._level + 1

    def dedent(self):
        self._level = self._level - 1

    def info(self, message):
        print "  " * self._level + message

    @contextmanager
    def mesure(self, what):
        t0 = time()
        yield
        self.info("%s performed in %s" % (what, time() - t0))

    def __getattr__(self, obj_name):
        """Load the different objects that will be used in the clustering process.
        
        For instance, vectorizers or big datasets will be loaded using this method.
        """
        filename = self.object_filename(obj_name)
        if os.path.isfile(filename):
            with open(filename) as f:
                self.info("unpickle %s" % obj_name)
                object = pickle.load(f)
        else:
            if hasattr(self, "_load_%s" % obj_name):
                object = getattr(self, "_load_%s" % obj_name)()
                with open(filename, "w+") as f:
                    self.info("save %s" % filename)
                    pickle.dump(object, f)
            else:
                raise NameError(obj_name)

        setattr(self, obj_name, object)
        return object

    def run(self):
        for username, docs in self.dataset():
            self.info("processing %s (%s docs)" % (username, len(docs)))
            self.indent()
            docs_tr, docs_2d = self.run_metrics(docs)
            for name, klass in self._clusters.items():
                cluster = klass().fit(docs_tr)
                if self._draw_pie:
                    self.draw_pie(cluster, name, username)
                elif self._draw_2d:
                    self.draw_2d(cluster, docs_2d, name, username)
            self.dedent()

    def draw_pie(self, cluster, algo, username):
        """Draw a pie shart with the percentages of documents in each cluster

        :param cluster: the clustering object
        :param username: the concerned username
        :param algo: the used algorithm (string)
        :param features: the name of the set of features used
        """
        features = self._features
        fig = pl.figure(figsize=(6, 6))
        fig.clf()
        labels = range(max(cluster.labels_) + 1)
        fracs = np.bincount(cluster.labels_)
        pl.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True, 
                colors=[c for c in 'bgrcmyk'])
        pl.title("distribution for %s (%s, %s)" % (username, features, algo))
        filename = os.path.join(OUTPUT_PATH, 
            "figures/pie_%s_%s_%s.png" % (features.lower(), algo, username))
        
        fig.savefig(filename)
        self.info("generated pie chart at %s" % filename)

    def draw_2d(self, cluster, X, algo, username):
        """Plot the results on a 2D chart.
        
        :param cluster: the clustering object containting the labels and the centers
        :param X: the dataset used (should be in 2 dimensions)
        :param username: The username is used for both the title of the figure and 
                         the name of the file.
        :param algo: the algorithm that was used for clustering
        """
        features = self._features
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
        filename = os.path.join(OUTPUT_PATH, 
            "figures/2d_%s_%s_%s.png" % (features.lower(), algo, username))
        fig.savefig(filename)
        self.info("generated 2d chart at %s" % filename)


class TextClusterer(Clusterer):

    _features = "text"

    def __init__(self, training_set=None, **kwargs):
        super(TextClusterer, self).__init__(**kwargs)
        self._training_set = training_set
        if not self._training_set:
            self._training_set = "newsgroup"

    def _load_vec(self):
        # equivalent to CountVectorizer + TfIdf
        return Vectorizer().fit(self.docs)

    def _load_pca(self):
        return RandomizedPCA(n_components=100, whiten=True).fit(
                self.vec.transform(self.docs))

    def _load_pca_2d(self):
        return RandomizedPCA(n_components=2, whiten=True).fit(
                self.vec.transform(self.docs))

    def _load_docs(self):
        if self._training_set == "newsgroup":
            self.info("extract the 20 newsgroup dataset")
            wide_dataset = fetch_20newsgroups()
            docs = [open(f).read() for f in wide_dataset.filenames]

        elif self._training_set == "docs":
            docs = [res['content'] for res in 
                    db.resources.find({'blacklisted': False, 
                                       'processed': True})]
        return docs

    def dataset(self):
        for username in list(db.users.find().distinct('username')):
            if self._particular_user and self.particular_user != username:
                continue

            # get all the resources for this user
            urls = db.views.find({"user.username": username}).distinct("url")
            if not urls:
                continue # if we don't have any url for this user, go to the next one!

            resources = list(db.resources.find({'url': {'$in': urls }, 
                'blacklisted': False, 'processed': True}))
            if not resources:
                continue
            
            # we are interested to work with the contents
            yield username, [res['content'] for res in resources]

    def run_metrics(self, docs):
        # fit the contents to the new set of features the PCA determined
        with self.mesure("reduce dataset dimensions to 100"):
            # transformed docs
            docs_tr = self.pca.transform(self.vec.transform(docs))

        # what we do have now is a matrix with 100 dimentions, which is not really 
        # useful for representation. Keeping this for later analysis is a good
        # thing so let's save this model for comparing profiles against resources
        # later

        # project X onto 2D
        with self.mesure("reduce dataset dimensions to 2"):
            docs_2d = self.pca_2d.transform(self.vec.transform(docs))

        return docs_tr, docs_2d


class ContextClusterer(Clusterer):

    _features = "context"

    def dataset(self):
        for username in db.users.distinct("username"):
            if self._particular_user and self._particular_user != username: continue 

            urls = db.views.find({"user.username": username}).distinct("url")
            resources = []
            if not urls: continue
                
            t0 = time()
            progress = ProgressBar(
                        widgets=["  building the matrix for %s" % username, 
                        Percentage(), Bar()])

            for url in progress(urls):
                # get the views related to this user and this url
                views = db.views.find({"user.username": username, "url": url})
                resources.append(self._get_features(views))

            self.info("matrix generation for %s took %s" % (username, time() - t0))
            yield username, np.array(resources)

    def _get_features(self, views):
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

        return row

    def run_metrics(self, docs):
        return docs, RandomizedPCA(n_components=2, whiten=True).fit(docs).transform(docs)


class ClusterUsers(TextClusterer):

    _features = "users-text"
    _draw_2d = False

    def dataset(self):
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
        yield "all", users_content

    def run_metrics(self, docs):
        return self.pca.transform(self.vec.transform(docs)), None


if __name__ == '__main__':
    clusterer = ClusterUsers()
    clusterer.run()
