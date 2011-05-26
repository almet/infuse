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
from flaskext.wtf import (Form, SubmitField, TextField, 
        PasswordField, ValidationError, Required, EqualTo)

# app imports
from db import *

# create the application, initialize stuff
app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_pyfile('settings.py')

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

@app.route("/connect/", methods=["POST"])
def are_credentials_valid():
    """Try to connect using the HTTP headers and return true or false"""
    authorized = True if _get_user() else False
    return jsonify(authorized=authorized)


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
