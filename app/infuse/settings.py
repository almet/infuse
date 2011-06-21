import os

# configuration
DEBUG = True
SECRET_KEY = "thisisnotsecret"
MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
MONGODB_DB = 'suggest'

BLACKLIST = ['facebook', 
        'google',
        'gmail',
        'amazon', 
        'twitter', 
        'delicious'
        ]

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
LIBS_PATH = os.path.join(BASE_PATH, "libs")
OUTPUT_PATH = os.path.join(BASE_PATH, "temp")

JAVA_CLASSPATH = "%(libs)s/boilerpipe-1.1-dev.jar:%(libs)s/nekohtml-1.9.13.jar:%(libs)s/xerces-2.9.1.jar" % {'libs': LIBS_PATH }
