Pysuggest
#########

Suggest is the implementation of a context-aware web recommendation system. It
is decoupled into two parts: a firefox plugin which retrieve the browsing
data and a server side script which expose an API and provide tools to extract
data, analyse them, cluster them and provide recommendations.

This application is the server side.

Web API
=======

A web API is intended to record users interaction with their browser. Users
need to be authentified to do so. This API only exposes ways to *record* data,
not to read it. Internally, the data is stored into a mongodb database.

User management
===============

Because users need to authenticate to send data through the API, there is
a need for user management. Users are abble to:

* create an account
* delete an account (along with all their data)
* access the recorded data about them
* modify they credentials

Data extraction
===============

Once the data have been recorded into the mongodb database, there is a need to
extract this information. This means:

* Gather the text of the HTML resources
* Extract text metrics from texts, idealy subjects / tag of words
* Transform opening / closing information to viewing sequences.

Profile extraction
==================

Extract profiles from the different information gathered at this point: text
subjects, browsing trees, geolocalisation. For each  user, determine the
different possible profiles, using clustering techniques.

Profile clustering
==================

Cluster profiles together.

Links ranking
=============

Given a number of heuristics, rank *the visited items*

Ranking prediction
==================

Uses collaborative filtering techniques to predict the rankings of unknown
items in the profiles clusters.
