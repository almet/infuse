import base64
import hashlib
from functools import wraps

from flask import *
from flaskext.wtf import (Form, SubmitField, TextField, 
        PasswordField, ValidationError, Required, EqualTo)

from mongokit import Connection, Document
from pymongo import binary

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


@connection.register
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


@connection.register
class TabRelation(Document):
    structure = {
            'tab_id': unicode, 
            'parent_id': unicode
    }

    use_dot_notation = True


@connection.register
class User(Document):
    structure = {
            'username': unicode, 
            'password': str
    }
    authorized_types = Document.authorized_types + [str]
    use_dot_notation = True


def need_authentication(func):
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
    """Check that the provided credentials are matching a real user.

    If not, return None
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


@app.route("/")
def index():
    return render_template("index.html", form=UserForm())

@app.route("/register/", methods=["GET", "POST"])
def register():
    """register a new user"""
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
    """record an event occured on the browser"""
    event = events.Event()
    for key, value in request.form.items():
        event[key] = value
    event.save()
    return "event added"


@app.route("/tab-relation/", methods=["POST"])
@need_authentication
def add_tabrelation(user):
    """record a relationship between tabs"""
    relation = tabs.TabRelation()
    relation['tab_id'] = request.form['tab_id']
    relation['parent_id'] = request.form['parent_id']
    relation.save()
    return "relation added"


@app.route("/debug")
def debug():
    return "debug done"

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
