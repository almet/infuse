"""Download a web content if not already downloaded.

The results are stored in a temporary database "resources".
"""
from urlparse import urlparse
from urllib2 import urlopen
import contextlib
import re
import fnmatch

from infuse.db import resources
from infuse.settings import BLACKLIST

def is_downloaded(url):
    """Return if the url have already been downloaded or not.

    :param url: the url to check
    """
    exists = resources.one({'url': url})
    return True if exists else False

def is_blacklisted(url):
    """Return is the url is blacklisted or not.

    :param url: the url to check
    """
    # split the url into domain and the rest
    parsed = urlparse(url)
    netloc = parsed.netloc
    scheme = parsed.scheme

    # only HTTP(S) is allowed
    if scheme not in ['http', 'https']:
        return True

    # check if the domain is blacklisted
    for pattern in BLACKLIST:
        regexp = re.compile(fnmatch.translate("*%s*" % pattern))
        if regexp.match(url):
            return True
    return False

def download(url):
    """Download an url into a document.

    Check that the url have not already been downloaded and that it is not a 
    blacklisted one.

    :param url: the url to download
    """
    # do not download if already downloaded or blacklisted
    if not is_blacklisted(url) and not is_downloaded(url):
        with contextlib.closing(urlopen(url)) as file:
            # read the content and store it into a resource document
            resource = resources.Resource()
            resource.url = url
            resource.content = file.read()
            resource.save()

def main():
    pass

if __name__ == '__main__':
    main()
