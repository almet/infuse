from collections import defaultdict

import numpy as np
from scikits.learn.cluster import KMeans
from scikits.learn.metrics.pairwise import euclidean_distances
from scikits.learn.svm import sparse as svm

from processors import TextProcessor, ClusterUsers, ContextProcessor
from drawing import draw_matrix, compare_pies
from settings import OUTPUT_PATH
import db

class Infuse(object):
    """The main object, conducting the operations"""

    def __init__(self):
        self.output_path = OUTPUT_PATH
        self._processor = None
        self._usernames = None
        self._rankings = None
        self._default_processor = lambda: TextProcessor(store_docs=True, 
                clusters={"kmeans": lambda: KMeans(5)})

    def compute_similarities(self):
        print "generated %s" % draw_matrix(
                self.text_users_similarity(), "users", self.output_path)
        print "generated %s" % draw_matrix(
                self.text_profiles_similarity(), "profiles", self.output_path)


    def text_profiles_similarity(self):
        """Compute and return similarity scores between profiles, based on text 
        features and KMeans clustering.
        """

        # Text (TF-IDF)
        processor = TextProcessor(store_docs=True, 
                clusters={'kmeans': lambda: KMeans(5)} )
        processor.run()
        
        # dictionary containing metrics for the profiles
        docs = []
        for username, cluster in processor.clusters["kmeans"].items():
            # for each cluster, build up a new dataset, we will then use it to 
            # compare the profiles
            for label in np.unique(cluster.labels_):
                # get only the documents with this label
                docs.append(" ".join([processor.docs[username][i] for i, val 
                    in enumerate(cluster.labels_ == label) if val]))

        features = processor.get_features(docs)
        self._processor = processor
        return euclidean_distances(features, features)


    def text_users_similarity(self):
        """Compute and return similarity scores between users, based on text features.
        """

        self._processor = ClusterUsers(store_docs=True)

        # for each user, we want to have a set of features representing it
        features = []
        for name, docs in self.processor.iterate():
            features = self.processor.get_features(docs)
            # there is only one tuple (name, docs) so we return here
            return euclidean_distances(features, features)


    def get_clusters(self):
        # ContextProcessor().run(); return
        pass

    def compare_pca(self):
        """Compare the clusters generated with different values for the dimensions
        of the PCA
        """

        processors = (
                TextProcessor(N=50, algorithms=["kmeans"]), 
                TextProcessor(N=100, algorithms=["kmeans"]), 
                TextProcessor(N=200, algorithms=["kmeans"])
        )

        users_cluster = defaultdict(list)
        for processor in processors:
            # don't use random centers for kmeans to be able to compare them
            processor._particular_user = "alexis"

            processor.run()
            for user, cluster in processor.clusters['kmeans'].items():
                users_cluster[user].append(np.bincount(cluster.labels_))

        for user, bincounts in users_cluster.items():
            compare_pies(bincounts, "compare_%s.png" % user, self.output_path)

    def get_best_matches(self, matrix, X, N):
        """Return the N best matches for X, regarding the similarity matrix.

        :param matrix: similarity matrix
        :param X: the row to consider
        :param N: the number of items to return
        """
        return matrix[X].argsort()[-N:][::-1]

    def collaborative_filtering(self, username, N=3):
        """Use collaborative filtering techniques to, given a set of users, a 
        similarity matrix between them and the rankings for the resources, return 
        a list of urls with their infered score.
        """

        user_id = self.usernames.index(username)

        # get the similarities bw users
        similarity = self.text_users_similarity()

        # get the N similar users
        similar_users = self.get_best_matches(
                similarity, user_id, N)

        weighted_ranks = {}
        # for each user, compute similarity * rankings (of each doc)
        for idx in similar_users:
            if idx != user_id: # we don't want to compute it for this user
                username = self.usernames[idx]
                # get the rankings for the resources
                rankings = self.rankings[username]
                weighted_ranks[username] = map(lambda x: (x[0], x[1] * similarity[user_id][idx]), rankings)

        # return the list

    @property
    def usernames(self):
        if not self._usernames:
            self._usernames = list(db.users.find().distinct('username'))
        return self._usernames

    @property
    def rankings(self):
        if not self._rankings:
            predictions = {}
            for username in self.usernames:
                # get the ranked items
                print "get the ranked urls"
                ranked_urls = db.views.find({'feedback':{'$ne':'none', '$exists': True}, 
                                              'user.username': username}).distinct('url')

                print "get the unranked urls"
                unranked_urls = db.views.find({'feedback':'none', 
                                                'user.username': username}).distinct('url')

                # we have views, we want resources
                print "get the ranked resources"
                ranked_resources = list(db.resources.find({
                    'url': {'$in': ranked_urls}, 
                    'blacklisted': False, 
                    'processed': True
                }))
                print "get the unranked resources"
                unranked_resources = list(db.resources.find({
                    'url': {'$in': unranked_urls},
                    'blacklisted': False, 
                    'processed': True
                }))

                labels = []
                print "get feedback"
                for resource in ranked_resources:
                    labels.append(int(db.views.find({
                        'url': resource['url'], 
                        'user.username': username
                    }).distinct('feedback')[0]))

                if ranked_resources and unranked_resources:
                    # get features from the content
                    print "get features from ranked dataset"
                    ranked_dataset = self.processor.get_features([i['content'] 
                        for i in ranked_resources], username)
                    print "get features for unranked dataset"
                    unranked_dataset = self.processor.get_features([i['content'] 
                        for i in unranked_resources], username)

                    classifier = svm.LinearSVC()
                    classifier.fit(ranked_dataset, labels)
                    predictions[username] = zip([i['url'] for i in ranked_resources],
                            labels)
                    predictions[username].extend(zip([i['url'] for i in unranked_resources], 
                            [classifier.predict(elem) for elem in unranked_dataset]))
                else:
                    predictions[username] = [(r['url'], 2.5) for r in unranked_resources]

            self._rankings = predictions
        return self._rankings

    @property
    def processor(self):
        if not self._processor:
            self._processor = self._default_processor()
        return self._processor
            

    def cluster_users(self):
        p = ClusterUsers()
        p.run()
        return p

    def run_processors(self):
        args = {'draw_2d': True, 'draw_pie': True}
        processors = (TextProcessor(**args), ContextProcessor(**args))
        for processor in processors:
            processor.run()

    def clean_users(self):
        """remove the users that don't have any document"""
        for user in list(db.users.find()):
            views = list(db.views.find({"user": user}))
            if len(views) == 0:
                db.users.remove(user)

def main():
    infuse = Infuse()
    infuse.clean_users()
    infuse.collaborative_filtering("alexis")

if __name__ == '__main__':
    main()
