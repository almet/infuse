"""This file contains the convertion logic from raw data to something which actually
means something valuable.

The raw data only contains data about the events that have been fired-up on the web
browser, so we do have informations like "this tab have been opened", "this tab have
been closed" and so on. This module convert this kind of information into 
information like "the user have watched this resource for XX seconds", 
"he watched it N times" etc.

The implementation is based on Map/Reduce functions that are directly executed on
the mongodb server.
"""

import datetime
import db

STARTERS = ["ready", "activate"]
ENDERS = ["deactivate", "close"]

class TemporaryView(object):
    def __init__(self):
        self.started = None
        self.finished = None
        self.event = None
        self.parent = None

    def duration(self):
        """return the duration, in seconds"""
        return (int(self.finished) - int(self.started)) / 1000

    def reset(self):
        self.__init__()

    def save(self):
        view = db.views.View()
        view.location = self.event['location']
        view.user = self.event['user']
        view.url = self.event['url']
        view.duration = self.duration()
        view.timestamp = int(self.event['timestamp'])
        
        # compute the time of day and the day of week
        dt = datetime.datetime.fromtimestamp(int(self.event['timestamp'])/1000)
        view.weekday = dt.weekday()

        if dt.hour <= 2:
            daytime = "0-3"
        elif dt.hour <= 4:
            daytime = "3-5"
        elif dt.hour <= 6:
            daytime = "5-7"
        elif dt.hour <= 8:
            daytime = "7-9"
        elif dt.hour <= 10:
            daytime = "9-11"
        elif dt.hour <= 12:
            daytime = "11-13"
        elif dt.hour <= 14:
            daytime = "13-15"
        elif dt.hour <= 16:
            daytime = "15-17"
        elif dt.hour <= 18:
            daytime = "17-19"
        elif dt.hour <= 20:
            daytime = "19-21"
        elif dt.hour <= 22:
            daytime = "21-23"
        else:
            daytime = "23-24"

        view.daytime = daytime
        view.save()

        # does a resource with this url exist ? if not, create one
        res = db.Resource.get_or_create(self.event['url'])
        if self.parent and self.parent not in res.parents:
            res.parents.append(self.parent)
        res.save()


def extract_views():
    """Extract information about the resources views.

    Reads the information from the Events collection (the raw information taken 
    directly from the browsers) and convert it into something usable in the views 
    collection."""
    def _mark_as_processed(event):
        if event:
            event['processed'] = True
            event.save()
    
    
    for id in db.events.distinct("tab_id"):
        # get all the events relative to this tab
        events = db.events.Event.find({"tab_id": id}).sort("timestamp")
       
        # init variables for the loop
        view = TemporaryView()
        previous_url = None

        for event in events:
            if event['type'] in STARTERS:
                view.started = event['timestamp']
                view.event = event

            elif view.started and event['type'] in ENDERS:
                view.finished = event['timestamp']

            # manage the relation between resources
            # Add the relation between the resources (different "ready" states 
            # with different urls on the same tab)
            if event['type'] == "ready":
                if previous_url:
                    # add information to the view
                    view.parent = previous_url

                previous_url = event['url']

            if view.finished:
                # save the event in the db
                # print("the resource %s have been viewed for %s seconds, the timestamp is %s" % (view.event['url'], view.duration(), view.event['timestamp']))
                view.save()
                view.reset()

                # mark the two events as processed
                _mark_as_processed(view.event)
                _mark_as_processed(event)


def reset():
    """Remove all the resources/views and mark all the events as not processed.

    This function is mainly used for test purposes"""
    for event in db.events.Event.find({'processed': True}):
        event.processed = False
        event.save()

    db.resources.drop()
    db.views.drop()


def summarize():
    """Summarize all the information and put them in a document containing information
    about:

        - The URL
        - The user
        - The number of views for this resource for this user
        - The overall view duration
        - The geolocations for those views
        - The time (period of the day, day of the week)

    This is only based on views. The content is not downloaded in this document but
    rather in a resource document to proceed. (Which is done in the download 
    module).
    """

reset()
#extract_views()
