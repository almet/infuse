# A context-aware web recommendation system 

.fx: title

---

## Recommendations?


> Recommendation systems attempts to recommend information items (movies, music,
  news, etc.) that are likely to be of the interest to the user
  
Wikipedia


---

## Goals

.fx: bigbullet

* Propose **new content** (web pages) based on submited information
* Do not have any user interaction with the data (e.g ranking)
* Find the best way to propose new contents using state of the art techniques

---

## No-Goals

.fx: bigbullet

* Implementing new clustering algorithms
* Having something that runs fast (even if it have been taken in consideration)

---

# Different recommendation systems

.fx: title

---

## Different recommendation systems

.fx: bigbullet

* Content based
* Colaborative filtering
* **Hybrids**

---

## Image explaining the different approachs

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

# Transformation of the data

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

* A lot of content is repeated and/or innacurate on webpages
* Text extraction is a way to find and remove it. (Boilerpipe paper)

---

# Finding profiles

.fx: title

---

## Different clustering techniques

* Supervised learning
* Unsupervised learning
* + images !

---

## Clustering the web pages / views

* By using text features
* By using view-related features (location, time of the day…)

---

## TF-IDF

* can get the most used terms
* need to train on a bigger dataset

---

## Views, the feature vectors problem

---

## Some graphics

---

## What is next?
