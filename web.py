from flask import *
from mongokit import Connection, Document

# configuration
DEBUG = True
SECRET_KEY = "thisisnotsecret"
MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017

# create the application, initialize stuff
app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('SETTINGS', silent=True)

connection = Connection(app.config['MONGODB_HOST'], app.config['MONGODB_PORT'])

class Event(Document):
    structure = {
            'type': unicode,
            'title': unicode,
            'url': unicode,
            'timestamp': unicode,
            'tab_id': unicode,
            'browser': unicode
    }

    use_dot_notation = True

class TabRelation(Document):
    structure = {
            'tab_id': unicode, 
            'parent_id': unicode
    }

    use_dot_notation = True

# connect the classers to the db
connection.register(Event)
connection.register(TabRelation)

# define shortcuts for later use
events = connection['suggest'].events
tabs = connection['suggest'].tabs

@app.route("/event/", methods=["POST"])
def add_event():
    """record an event occured on the browser"""
    event = events.Event()
    for key, value in request.form.items():
        event[key] = value
    event.save()
    return "event added"

@app.route("/tab-relation/", methods=["POST"])
def add_tabrelation():
    """record a relatioship between tabs"""
    relation = tabs.TabRelation()
    relation['tab_id'] = request.form['tab_id']
    relation['parent_id'] = request.form['parent_id']
    relation.save()
    return "relation added"


@app.route("/debug")
def debug():
    from ipdb import set_trace
    set_trace()
    return "debug done"

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
