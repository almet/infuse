import datetime

from mongokit import Connection, Document
from pymongo import binary

from settings import *

connection = Connection(MONGODB_HOST, MONGODB_PORT)

events = connection[MONGODB_DB].events
tabs = connection[MONGODB_DB].tabs
users = connection[MONGODB_DB].users
resources = connection[MONGODB_DB].resources
views = connection[MONGODB_DB].views

@connection.register
class User(Document):
    structure = {
            'username': unicode, 
            'password': str
    }
    authorized_types = Document.authorized_types + [str]
    use_dot_notation = True


@connection.register
class Event(Document):
    structure = {
            'type': unicode,
            'title': unicode,
            'url': unicode,
            'timestamp': unicode,
            'tab_id': unicode,
            'browser': unicode,
            'user': User,
            'location': list,
            'processed': bool,
    }

    authorized_types = Document.authorized_types + [tuple]
    use_dot_notation = True


@connection.register
class TabRelation(Document):
    structure = {
            'tab_id': unicode, 
            'parent_id': unicode,
            'user': User,
    }

    use_dot_notation = True


@connection.register
class Resource(Document):
    """Represents a url and its content"""
    structure = {
            'url': unicode,
            'content': unicode,
            'parents': list,
            'processed': bool,
            'date': datetime.datetime,
    }

    authorized_types = Document.authorized_types + [str]
    default_values = {'processed': False, 'parents': [], 
            'date': datetime.datetime.now()}
    use_dot_notation = True

    @staticmethod
    def get_or_create(url):
        """Return a resource, create one if one with the specified url doesnt 
        already exists.

        The returned resource is not persisted in db when returned, so there is a
        need to call save() on it after using it.

        :param url: the url of the resource
        """
        resource = resources.Resource.one({'url': url})
        if not resource:
            resource = resources.Resource()
            resource.url = url
            resource.processed = False

        return resource


@connection.register
class View(Document):
    structure = {
            'url': unicode, 
            'user': User,
            'duration': int, 
            'location': list,
            'timestamp': int,
            'daytime': str,
            'weekday': int,
    }

    authorized_types = Document.authorized_types + [str]
    use_dot_notation = True
