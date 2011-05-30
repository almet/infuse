from mongokit import Connection, Document
from pymongo import binary

from infuse.settings import *

connection = Connection(MONGODB_HOST, MONGODB_PORT)

events = connection[MONGODB_DB].events
tabs = connection[MONGODB_DB].tabs
users = connection[MONGODB_DB].users
contents = connection[MONGODB_DB].contents

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
            'location': tuple
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
class Content(Document):
    """Represents a url and its content"""
    structure = {
            'url': str,
            'content': unicode
    }

    authorized_types = Document.authorized_types + [str]
    use_dot_notation = True
