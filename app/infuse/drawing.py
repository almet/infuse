import os
import pylab as pl
import numpy as np

def draw_pie(cluster, algo, username, features, output_path):
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
    filename = os.path.join(output_path, 
        "figures/pie_%s_%s_%s.png" % (features.lower(), algo, username))
    
    fig.savefig(filename)
    return filename

def draw_2d(cluster, X, algo, username, features, output_path):
    """Plot the results on a 2D chart.
    
    :param cluster: the clustering object containting the labels and the centers
    :param X: the dataset used (should be in 2 dimensions)
    :param username: The username is used for both the title of the figure and 
                     the name of the file.
    :param algo: the algorithm that was used for clustering
    :param features: the set of features that was used
    :param output_path: the path to output to
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
        # x and y axis are computed from a PCA and so can't be labeled
        ax.plot(X[my_members, 0], X[my_members, 1], col)

        # plot the centers
        ax.plot(cluster_center[0], cluster_center[1], 'o' if col[1] == '.' else col[1], 
                markerfacecolor=col[0], markeredgecolor='k', markersize=14)

        ax.set_title('%s: %s clusters using %s' % (algo, n_clusters_, features))
    filename = os.path.join(output_path, 
        "figures/2d_%s_%s_%s.png" % (features.lower(), algo, username))
    fig.savefig(filename)
    return filename


def compare_pies(series, name, output_path):
    """Compare the given series using pie representation"""
    fig = pl.figure()

    len(series)
    for idx, serie in enumerate(series):
        ax = fig.add_subplot(1, len(series), 1 + idx)
        ax.pie(serie, autopct='%1.1f%%', shadow=True, colors=[c for c in 'bgrcmyk'])

    filename = os.path.join(output_path, 
        "figures/compare_pies_%s.png" % name)
    fig.savefig(filename)
    return filename


def draw_matrix(similarity, filename, output_path):
    """Draw a matrix into the specified filename"""
    fig = pl.figure()
    sub = fig.add_subplot(111)
    sub.matshow(similarity)
    filename = os.path.join(output_path, "figures/sim_matrix_%s.png" % filename)
    fig.savefig(filename)
    return filename
