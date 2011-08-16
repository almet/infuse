import os
import pickle
from collections import defaultdict
from operator import itemgetter

from scikits.learn.cluster import KMeans, MeanShift, estimate_bandwidth
from scikits.learn.metrics.pairwise import euclidean_distances
from scikits.learn.svm import sparse as svm
import numpy as np

import db
from settings import OUTPUT_PATH
from utils import mesure
from drawing import draw_matrix


def get_views_features(views):
    features = []

    with mesure("building the vector space", indent=1):
        # build the vector space
        for view in views:
            features.append([
                # lat, long
                float(view['location'][0]), float(view['location'][1]), 
                int(view['duration']),
                int(view['daytime']),
                # split the day in 6 bits of 4 hours.
                int(view['daytime']) / 4, 
                int(view['weekday'])
            ])
    return features


def get_profiles_similarity(usernames, N):
    """
    Return a matrix of similarity between the users.

    :usernames:
        The list of usernames in the system

    :N: 
        the number of profiles to find for each user
    """

    # all the documents per profile will be stored in this variable
    doc_profiles = []

    # all the urls for each profiles will be put in this array
    urls = []

    # For each user, get his views
    for username in usernames:
        print "processing %s" % username

        # don't use generators are we want to access it multiple times, so we
        # actually need to store it in memory
        views = list(db.views.find({"user.username": username, 'url': {
            '$nin': list(db.resources.find({'blacklisted': True}).distinct('url'))
        }}))

        features = get_views_features(views)

        # Run a clustering algorithm on the view
        np_features = np.array(features)
        #bandwidth = estimate_bandwidth(np_features, quantile=0.3)
        #algo = MeanShift(bandwidth=bandwidth).fit(np_features)

        # The distribution from the KMeans algorithm is better because we get
        # more balanced clusters. MeanShift comes with a lot of clusters with 
        # less than 2 elements.
        with mesure("clustering the context to find %s profiles" % N, indent=1):
            algo = KMeans(N).fit(np_features)

        # for each cluster, get the matching views
        # this means iterating N times (where N is the number of cluster found)
        for label in np.unique(algo.labels_):

            profile_urls = []
            for i, matches in enumerate(algo.labels_ == label):
                view = views[i]
                if matches and view['url'] not in profile_urls:
                    profile_urls.append(view['url'])

            # save the urls of this profile for later use
            urls.append(profile_urls)
            
            resources = db.resources.find({
                'url': {'$in': profile_urls}, # get the resources for those urls
                'blacklisted': False, 'processed': True})

            # Append the contents for this profile together
            doc_profiles.append(" ".join([r['content'] for r in resources]))

    # train the vectorizer on a big and sparse set of documents
    # the vectorizer is loaded from disk to avoid recomputing it each time
    with open(os.path.join(OUTPUT_PATH, "pickle", "vecnewsgroup.pickle")) as f:
        vec = pickle.load(f)
    
    # Same for the principal component analysis (PCA)
    with open(os.path.join(OUTPUT_PATH, "pickle", "pca100-newsgroup.pickle")) as f:
        pca = pickle.load(f)

    # At this stage, all the documents are stored into memory, sometimes
    # more than once for each resource. We want to vectorize them all and thus
    # it can take some time.
    with mesure("vectorizing %s profiles" % len(doc_profiles)):
        vec_profiles = pca.transform(vec.transform(doc_profiles))

    # Compute their similarity score
    return euclidean_distances(vec_profiles, vec_profiles), urls


def collaborative_filtering(usernames, similarity, rankings, urls, N):
    """Do the collaborative filtering for the given usernames, rankings and 
    similarity between profiles.

    :usernames:
        the list of usernames in the system

    :similarity:
        The similarity matrix for all the profiles

    :rankings:
        An array of [username][url] = score

    :N:
        the number of profiles used per user
    """

    # XXX Eventually split here.

    # draw the matrix for later analysis
    draw_matrix(similarity, "final_kmeans", OUTPUT_PATH)

    # For each profile, get the best matches. 
    user_id = 0
    weighted_ranks = defaultdict(dict)
    # p_sim is for "profile similarity"
    for idx, p_sim in enumerate(similarity):
        if idx % N == 0:
            # we iterated over all the profiles for this user
            user_id = user_id + 1

        # ignore the profiles from the same user
        matching_profiles = [i for i in p_sim.argsort()[::-1] 
                if i < N * (user_id - 1) or i > N * (user_id)][:10]

        # for all the matching profiles, get the related urls, and construct a 
        # list of weighted ranks, in the form url, rank
        for profile_id in matching_profiles:
            username = usernames[profile_id / N]
            # get the urls for this profile
            profile_urls = urls[profile_id]
            
            # use the collaborative filtering technique to weigth all the urls
            # with the similarity scores between profiles
            for url in profile_urls:
                weighted_ranks[idx][url] = \
                    rankings[username][url] * p_sim[profile_id]
    
    recommendations = defaultdict(dict)
    # All the urls have been ranked, now get the M best ones in total
    for starting_profile in range(len(weighted_ranks))[::N]:
        user_rankings = {}
        for profile_id in range(starting_profile, starting_profile + N):
            profile_rankings = weighted_ranks[profile_id]
            for url in (user_rankings.viewkeys() & profile_rankings.viewkeys()):
                profile_rankings[url] = profile_rankings[url] + user_rankings[url]
            user_rankings.update(profile_rankings)
        
        user_rankings = user_rankings.items()
        sorted(user_rankings, key=itemgetter(1), reverse=True)
        recommendations[usernames[starting_profile / N]] = user_rankings

    return recommendations

def get_rankings(usernames):
    """Return the rankings for the given list of usernames.

    If rankings exists for some views of an url, the other ones are infered from 
    them. Otherwise, they are direcly mapped to simple heuristic rules.

    :usernames:
        The list of usernames to work with.
    """

    def _get_views_from_resources(resources, username):
        views = []
        for resource in resources:
            views.append(db.views.find_one({
                "url": resource['url'], 
                "user.username": username
            }))
        return views

    predictions = {}
    for username in usernames:
        # get the ranked items
        print "get the ranked urls"
        ranked_urls = db.views.find(
                {'feedback':
                    {'$ne':'none', '$exists': True}, 
                    'user.username': username}
                ).distinct('url')

        print "get the unranked urls"
        unranked_urls = db.views.find(
                {'feedback':'none', 
                 'user.username': username}).distinct('url')

        # we have views, convert them to resources
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

        # Only the first view of an url is selected here and used as label.
        # XXX Could it be useful to evaluate multiple times those resources
        # for all the different views?
        # This would mean to have multiple times different views of a resource,
        # mapped to the same score.
        for resource in ranked_resources:
            labels.append(int(db.views.find({
                'url': resource['url'], 
                'user.username': username
            }).distinct('feedback')[0]))

        if ranked_resources and unranked_resources:
            # get features for the resources

            print "get features from ranked dataset"
            ranked_dataset = get_views_features(
                    _get_views_from_resources(list(ranked_resources), username))

            print "get features for unranked dataset"
            unranked_dataset = get_views_features(
                    _get_views_from_resources(list(unranked_resources), username))

            # Classify the unranked resources.
            # Here, the SVC is learning a model from the known rankings and apply
            # it to the unknown ones to return predictions on what would be the 
            # score for them.
            classifier = svm.LinearSVC()
            classifier.fit(ranked_dataset, labels)
            temp_predictions = zip(
                    [i['url'] for i in ranked_resources],
                    labels
            )
            temp_predictions.extend(zip(
                [i['url'] for i in unranked_resources], 
                classifier.predict(unranked_dataset)
            ))
            predictions[username] = dict(temp_predictions)
        else:
            # happens only when the user didn't gave any feedback about
            # the views he made (that's the case for the majority of the users
            # Use simple heuristics to get the ranks

            # We need to get some information such as the average duration of
            # a visit ...
            #duration_avg #FIXME

            for resource in unranked_resources:
                # get all the views of this resource
                # 
                # compute the diff with the avg duration
                # was the resource viewed a lot?
                pass
            predictions[username] = dict([(r['url'], 2.5) for r in 
                    unranked_resources])

            # FIXME handle other cases, when we only have ranked resources

    return dict(predictions)

def main():
    usernames = list(db.users.find().distinct('username'))
    N = 10

    # try with 10 profiles per user
    similarity, urls = get_profiles_similarity(usernames, N) 

    # get the rankings for all the users
    rankings = get_rankings(usernames)

    # get the recommendations
    recommendations = collaborative_filtering(usernames, similarity, rankings, urls, N)

    return recommendations

if __name__ == "__main__":
        
    recommendations = main()
    from ipdb import set_trace; set_trace()
