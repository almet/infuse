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

LIBS_PATH = os.sep.join([os.path.dirname(os.path.abspath(__file__)),
                              "../../libs"])
JAVA_CLASSPATH = "%(libs)s/boilerpipe-1.1-dev.jar:%(libs)s/nekohtml-1.9.13.jar:%(libs)s/xerces-2.9.1.jar" % {'libs': LIBS_PATH }
