# Research background (the theoretical framework)

.fx: title

---

## Recommendation systems

.fx: fullimage

* In all recommendation systems you have the concepts of **users** and **items**

![](images/ranking.png)

---

## Techniques

.fx: bigbullet

Different ways to make recommendations

* **Content-based**: creating a profile and comparing items agains this profile
* **Colaborative filtering**: creating a similarity matrix between users and using 
  it to weight the items
* **Hybrids** systems

---

## Content based systems

.fx: fullimage

![](images/content-based.png)

---

## Collaborative filtering

.fx: fullimage

![](images/collaborative-filtering.png)

---

## Hybrid systems

.fx: fullimage

![](images/hybrid.png)

---

# Clustering

.fx: title

---

## Clustering

.fx: centerquote

> Cluster analysis or clustering is the assignment of a set of observations into 
  subsets (called clusters) so that observations in the same cluster are similar 
  in some sense.

---

## Using context

.fx: bigbullet

* The collaborative filtering approach is based on *similarity scores* between **users**.
* What if we find and isolate a number of **profiles** per user and compute the similarity
  scores between all those profiles?

---

## Different algorithms

.fx: bigbullet

* Hierarchical clustering (using dendograms)
* Partitional clustering
    * KMeans
    * MeanShift

---

## Distance metrics

.fx: bigbullet

* Euclidean distance
* Manathan metric
* others
