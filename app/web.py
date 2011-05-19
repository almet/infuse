"""This is a tiny application to record information provided by a browser addon.
It allows to:

    * Trace the events that occured on the browser side (open tab, close tab,
      content loaded etc.)
    * Create a new user
    * Record the relation between tabs (ans so allows to detect tabs trees easily)

The information is recorded in a MongoDB database for later use.

The original implementation uses a firefox addon as a client, but this server side 
script does not rely on specific firefox features.
"""
# python stdlib imports
import base64
import hashlib
from functools import wraps

# flask imports
from flask import *
from mongokit import Connection, Document
from pymongo import binary
from flaskext.wtf import (Form, SubmitField, TextField, 
        PasswordField, ValidationError, Required, EqualTo)

# configuration
DEBUG = True
SECRET_KEY = "thisisnotsecret"
MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017

# create the application, initialize stuff
app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('SETTINGS', silent=True)

# some mongdb related initialisations
connection = Connection(app.config['MONGODB_HOST'], app.config['MONGODB_PORT'])
events = connection['suggest'].events
tabs = connection['suggest'].tabs
users = connection['suggest'].users

# forms
class UserForm(Form):
    username = TextField("Username", validators=[Required()])
    password = PasswordField("Password", validators=[Required()])
    password2 = PasswordField("Again", 
            validators=[Required(), EqualTo("password")])
    submit = SubmitField("create an account")

    def validate_username(self, field):
        """Check that the username is not already defined"""
        user = users.one({'username': field.data})
        if user:
            raise ValidationError("This username is already used")


# mongodb documents
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


# utils
def need_authentication(func):
    """Decorator for view functions. Only authorise to enter the view if an user is 
    authenticated by the HTTP Authorised headers and there actually is an user
    matching those credentials.

    Return a 401 HTTP error otherwise.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        user = _get_user()
        if user:
            kwargs['user'] = user
            return func(*args, **kwargs)
        else:
            abort(401)
    return wrapper


def _get_user():
    """Return the user authentified by the Authorization header.
    
    Flask exposes the request object as a global object by design, so there is no
    need to pass the request object here.
    
    If an user matching the given credentials is given, return it, otherwise return
    None.
    """
    if "Authorization" in request.headers:
        authstring = base64.decodestring(
                request.headers["Authorization"].split(" ")[1])
        username, password = authstring.split(":")
        m = hashlib.md5()
        m.update(password)
        user = users.one({'username': username, 'password': m.hexdigest()})
        return user
    return None


# views
@app.route("/")
def index():
    return render_template("index.html", form=UserForm())

@app.route("/register/", methods=["GET", "POST"])
def register():
    """Register a new user. """
    form = UserForm()
    if request.method == 'POST':
        if form.validate():
            user = users.User()
            user.username = form.username.data
            m = hashlib.md5()
            m.update(form.password.data)
            user.password = m.hexdigest()
            user.save()
            flash("You user have been created successfully")
            return redirect(url_for("install"))
    return render_template("register.html", form=form)

@app.route("/install/")
def install():
    return render_template("install.html")

@app.route("/event/", methods=["POST"])
@need_authentication
def add_event(user):
    """Record an event that occured on the browser.
    
    Record all the informations related to the events such as:

        * the type of the event
        * the title of the tab concerned
        * the time of the event
        * the tab identifier
        * the browser used
        * the related user
        * the latitude and longitude

    Those information are stored "as-is" and saved for later processing.
    """
    event = events.Event()
    for key in ('type', 'title', 'url', 'timestamp', 'tab_id'):
        event[key] = request.form[key]
    event.location = (request.form['lat'], request.form['long'])
    event.user = user
    event.save()
    return "event added"


@app.route("/tab-relation/", methods=["POST"])
@need_authentication
def add_tabrelation(user):
    """Record relationships between tabs.
    
    The identifier used here are unique and thus there is no need to store the
    user. However, this information is recorded so it is easy to retrieve
    information quicker.

    The Tab identifiers are the same as the one used in the Events.
    """
    relation = tabs.TabRelation()
    relation['tab_id'] = request.form['tab_id']
    relation['parent_id'] = request.form['parent_id']
    relation.user = user
    relation.save()
    return "relation added"


@app.route("/debug")
def debug():
    """Only used for debug purposes. This will not work on the production server.
    """
    from ipdb import set_trace; set_trace()
    return "debug done"

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
