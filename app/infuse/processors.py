from itertools import cycle
import sys
import os
import pickle
from time import time
from collections import defaultdict
from functools import partial

from progressbar import ProgressBar, Bar, Percentage
from scikits.learn.feature_extraction.text import Vectorizer, TfidfTransformer
from scikits.learn.decomposition import RandomizedPCA
from scikits.learn.cluster import (MeanShift, KMeans, AffinityPropagation, 
                                   SpectralClustering)
from scikits.learn.datasets import fetch_20newsgroups
from scikits.learn.metrics.pairwise import euclidean_distances

import numpy as np
import pylab as pl

import db
from settings import OUTPUT_PATH
from utils import mesure
from drawing import draw_pie, draw_2d, draw_matrix, compare_pies

"""
This module defines classes able to extract information from the stored documents.

It has been created mainly to provide structure and reusable bits of code, in
order to stick to the DRY (Don't repeat yourself) principle.

This module introduces a super class "Processor", which aims to be a base class
for clusterer classes.
"""

class Processor(object):
    """Base class to easily use clustering algorithms on various datasets.

    The basic usage for this class is to subclass it and to redefine the following
    methods:

    * iterate(): yields an (username, documents) tuple that will be processed
    * get_features(): run various metrics on the yielded documents. Returns the
                      features for each doc.

    Additionaly, you can configure the execution flow by modifying class attributes
    defined below.

    One of the interesting features provided by this class is the ability to pickle
    and unpickle (load and read from file) big objects. In order to use this, you 
    need to define a "load_*name*" method, returning the object to load/save.

    This object should *not* start with an underscore, you can tweak its parameters
    using the _*name*_params class attribute.
    """

    _features = "all"
    _clusters = { "kmeans": KMeans, "meanshift": MeanShift }
    _draw_pie = False
    _draw_2d = False
    
    def __init__(self, particular_user=None, store_docs=False, algorithms=(), 
            output_path=None, **kwargs):
        self._particular_user = particular_user
        self._level = 0
        self.clusters = defaultdict(dict)
        self.algorithms = algorithms
        self.docs = {}
        self.store_docs = store_docs
        self.output_path = output_path or OUTPUT_PATH

        for key, value in kwargs.items():
            setattr(self, "_%s" % key, value)

        # init the mesure utility
        self.mesure = partial(mesure, print_f=self.info)

    def get_object_params(self, obj_name):
        """Return the object parameters, to be given to the loader"""
        if hasattr(self, "_%s_params" % obj_name):
            params = getattr(self, "_%s_params" % obj_name)
        else:
            params = []
        return params

    def object_filename(self, obj_name):
        """Return the path of the pickle file to load/write objects to/from"""
        params = self.get_object_params(obj_name)
        hash = "-".join([str(p) for p in params]) if params else ""

        return os.path.join(OUTPUT_PATH, "pickle/%s%s.pickle" % (obj_name, hash))

    def indent(self):
        """indent the messages"""
        self._level = self._level + 1

    def dedent(self):
        """dedend the messages"""
        self._level = self._level - 1

    def info(self, message, indent=False):
        """Print a message to stdout"""
        if indent:
            self.indent()
        print "  " * self._level + message
        if indent:
            self.dedent()

    def __getattr__(self, obj_name):
        """Load the different objects that will be used in the clustering process.
        
        For instance, vectorizers or big datasets will be loaded using this method.
        """
        if obj_name.startswith("_"):
            raise AttributeError("obj_name")

        filename = self.object_filename(obj_name)
        if os.path.isfile(filename):
            with open(filename) as f:
                self.info("unpickle %s" % obj_name)
                object = pickle.load(f)
        else:
            if hasattr(self, "_load_%s" % obj_name):
                self.info("loading %s" % obj_name)
                object = getattr(self, "_load_%s" % obj_name)\
                        (*self.get_object_params(obj_name))
                with open(filename, "w+") as f:
                    self.info("save %s" % filename)
                    pickle.dump(object, f)
            else:
                raise NameError(obj_name)

        setattr(self, obj_name, object)
        return object

    def run(self):
        """Main loop."""
        with self.mesure("starting %s" % self.__class__, indent=True):
            for what, docs in self.iterate():
                # store the docs
                if self.store_docs:
                    self.docs[what] = docs
                self.info("processing %s (%s)" % (what, len(docs)))
                self.indent()
                docs_tr = self.get_features(docs, what)
                if self._draw_2d and hasattr(self, "get_features_2d"):
                    docs_2d = self.get_features_2d(docs, what)

                for name, klass in self._clusters.items():
                    if self.algorithms and name not in self.algorithms:
                        continue
                    cluster = klass().fit(docs_tr)
                    self.clusters[name][what] = cluster
                    if self._draw_pie:
                        filename = draw_pie(cluster, name, what, self._features,
                                self.output_path)
                        self.info("generated pie chart at %s" % filename)
                    elif self._draw_2d:
                        filename = draw_2d(cluster, docs_2d, name, what,
                            self._features, self.output_path)
                        self.info("generated 2d chart at %s" % filename)
                self.dedent()



class TextProcessor(Processor):
    """Find profiles for each user, using different clustering techniques.

    The set of features used here is closely related to text. The algorithm does
    the following:

    * Computes the TF-IDF scores for a wide dataset (the 20 newsgroup dataset)
    * For each user's documents, get the weights of the words defined in the 20 
      newsgroup dataset.
    * Reduce the dimension to N (to lower the computation time)
    * Uses different clustering algorithms to find user clusters
    * Output various graphics about the found clusters
    """

    _features = "text"

    def __init__(self, training_set=None, N=100, **kwargs):
        super(TextProcessor, self).__init__(**kwargs)
        self._training_set = training_set
        self._pca_params = [N,]

        if not self._training_set:
            self._training_set = "newsgroup"

    def _load_vec(self):
        # equivalent to CountVectorizer + TfIdf
        return Vectorizer().fit(self.docs)

    def _load_pca(self, N):
        return RandomizedPCA(n_components=N, whiten=True).fit(
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

    def iterate(self):
        for username in list(db.users.find().distinct('username')):
            if self._particular_user and self._particular_user != username:
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

    def get_features(self, docs, username=None):
        # fit the contents to the new set of features the PCA determined
        with self.mesure("reduce dataset dimensions to %s" % self._pca_params[0]):
            # transformed docs
            docs_tr = self.pca.transform(self.vec.transform(docs))
        return docs_tr

    def get_features_2d(self, docs, username=None):
        # project X onto 2D
        with self.mesure("reduce dataset dimensions to 2"):
            docs_2d = self.pca_2d.transform(self.vec.transform(docs))
        return docs_2d


class ContextProcessor(Processor):
    """Uses information coming from the context to find clusters.

    For each of the user, for each of its url, create features representing
    the different contextual information such as location, tim of the day, 
    day of the week etc.

    As such contextual information are provided in lists - One item for each access
    (view) to the resource (url) - there is a need to flatten this data.

    Flattening is due using different mathematic operations such as average, mean,
    etc.

    The created dataset is then given to different algorithms and 2d and pie graph
    representations of the clusters are generated.
    """

    _features = "context"

    def iterate(self):
        for username in db.users.distinct("username"):
            if self._particular_user and self._particular_user != username: continue 

            urls = db.views.find({"user.username": username}).distinct("url")
            resources = []
            if not urls: continue
            yield username, urls

    def _process_views(self, views):
        """Returns a set of features from a set of views of an url"""
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

    def get_features(self, urls, username):
        t0 = time()
        progress = ProgressBar(
                    widgets=["  building the matrix for %s" % username, 
                    Percentage(), Bar()])

        features_dataset = []
        for url in progress(urls):
            # get the views related to this user and this url
            views = db.views.find({"user.username": username, "url": url})
            features_dataset.append(self._process_views(views))

        self.info("matrix generation for %s took %s" % (username, time() - t0))
        return np.array(features_dataset)


class ClusterUsers(TextProcessor):
    """Clusters all the users between them, *without* finding profiles as a first
    step.

    All the text contents are appenned to each other and TF-IDF is then computed
    for each user. It is then projected on the wider 20 newsgroup vector space 
    reduced to N dimensions (where N is specified in the constructor).
    """

    _features = "users-text"

    def __init__(self, **kwargs):
        """N is the number of dimensions to reduce to. (Default is 100)"""
        super(ClusterUsers, self).__init__(**kwargs)
        self._users_labels = []

    def iterate(self):
        users_content = []
        for username in list(db.users.find().distinct('username')):
            # get all the resources for this user
            urls = db.views.find({"user.username": username}).distinct("url")
            if not urls:
                continue # if we don't have any url for this user, go to the next one!

            resources = list(db.resources.find({'url': {'$in': urls }, 
                'blacklisted': False, 'processed': True}))
            if not resources:
                continue
            self.info("aggregating documents from %s (%s docs)" % (username, len(resources)))

            # get the docs content and names
            self._users_labels.append(username)
            users_content.append(" ".join([res['content'] for res in resources]))
        yield "users", users_content


class TextAndContextProcessor(TextProcessor, ContextProcessor):
    """Put the text and context features together and cluster the results using
    them both.
    """

    def iterate(self):
        """Return a set of features from """

        for username in db.users.distinct("username"):
            if self._particular_user and self._particular_user != username: continue 

            urls = db.views.find({"user.username": username}).distinct("url")
            if not urls: continue

            resources = list(db.resources.find({'url': {'$in': urls }, 
                'blacklisted': False, 'processed': True}))
            if not resources: continue
                
            progress = ProgressBar(
                        widgets=["  building the matrix for %s" % username, 
                        Percentage(), Bar()])

            for url in progress(urls):
                # get the views related to this user and this url
                views = db.views.find({"user.username": username, "url": url})
                context = self._get_features(views)
                # add text here !

            
            # we are interested to work with the contents
            yield username, [res['content'] for res in resources]
