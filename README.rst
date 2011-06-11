Infuse
######

Infuse is the implementation of a context-aware web recommendation system. It
is decoupled into two parts: a firefox plugin which retrieve the browsing
data and a server side script which expose an API and provide tools to extract
data, analyse them, cluster them and provide recommendations.

This is the server side.

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

Those steps are done in the following scripts:

* Gathering the text of the HTML resources + extracting metrics is done in the
  infuse.download module. It uses a python/java bridge to use a java tool able
  to transform HTML content into text content.
  If you run download.py a queue will listen for new urls to retrieve
  / transform / extract.
* Converting the events that fired up on the browser is done in the
  infuse.convert module. Each time a new resource is detected, it is sent
  through the queue to the download module.

Here is a simple scheme explaining the overall process::

       Browser ---------------------> Server
                                        |
                          ---------> MongoDB <-----------
                          |                             |
     Boilerpipe <--- Download <----- RabbitMQ <--- Convert events
       (java)        resources                        to views
   

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

Given a number of heuristics, rank *the visited items*. An interface provides
a way for users to give explicit feedback. As users are not forced to give
feedback, this step is not mendatory, but will be proposed to them.

It is possible to provide feedback by going through
http://infuse.notmyidea.org/feedback/

Ranking prediction
==================

Uses collaborative filtering techniques to predict the rankings of unknown
items in the profiles clusters.


Installation
============

Most of the dependencies can be installed automatically using the following
command::

    $ pip install -r requirements.txt

However, it will be needed to install manually mongodb (the server), rabbitmq
(the queue manager) and jpype (a python/java bridge). Similarly, you will need
to install numpy before running `pip install` as `scikits.learn` depends on it.
