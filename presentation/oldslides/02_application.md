# Application (the practice)

.fx: title

---

## Overall architecture

![](images/architecture.png)

---

# Extracting information from the browser

.fx: title

---

## What information can we use?

.fx: bigbullet

* Tabs
* Web pages
* Views
* Geolocation

---

## A firefox plugin

… and a tiny website to record the events.

.fx: bigbullet

* Authentication on a central server
* Sending information
* Storing all actions in a database:
    * opening / closing tabs
    * loading content to an existing tab
    * relation between tabs

---

## Some statistics

.fx: bigbullet

Since the end of May:

* **25** registered users, **12** sending data regularly
* **95180** events triggered and saved
* **8972** web pages viewed and scrapped by the system

---

# Preparing the data

.fx: bigbullet

* Converting events to a number of **views**.
* Each view have different properties:
    * time of the day the view was made
    * duration of the viewing
    * location of the view (latitude / longitude)
    * the viewed url
    * the related tab tree

---
## HTML download and convertion.

.fx: bigbullet

* Downloads web pages
* Removes unwanted text
* Detect the used language (NLP / Metadata)
* Convert it to text (not HTML)
* Blacklist some websites

---

# Finding profiles

.fx: title

---

## Clustering the web pages / views

.fx: bigbullet

* By using text features
* By using view-related features (location, time of the day…)

---

## Finding text-related contexts

.fx: bigbullet

TF-IDF

* can get the most used terms
* need to train on a bigger dataset
* provides clusters based on text usage

---

## Finding text-related contexts (2)

.fx: bigbullet

* Sparse data
* A lot of dimensions (107129 on the 20 newsgroup corpus)
* Need to reduce the number of dimensions

---

## Dimentionality reduction

.fx: bigbullet

* Principal Component Analysis (PCA) keeps only the most significant singular vectors
  to project the data to a lower dimensional space.
* In other words: reduce the number of dimensions, keep the variance
* Allow to be less computionary intensive
* Useful to project the data on 2D

---

## Text clusters

.fx: fullimage

![](images/kmeans_tfidf.png)

---

## Text clusters

.fx: fullimage

![](images/meanshift_tfidf.png)

---

## Text clusters

.fx: fullimage

![](images/kmeans_tfidf_pie.png)

---

## Text clusters

.fx: fullimage

![](images/meanshift_tfidf_pie.png)
---

## Finding navigation contexts

.fx: bigbullet

Clusters the resources based on:

* The number of times a resource have been made
* From where it have been viewed
* The time of the day
* The day of the week

---

## Some graphics

TF-IDF
![](images/architecture.png)
---

## What is next?
