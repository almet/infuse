"""Download a web content if not already downloaded.

The results are stored in a temporary database "resources".
"""
from urlparse import urlparse
import urllib2
import contextlib
import re
import fnmatch
from threading import Thread
import sys
import multiprocessing

import chardet
from progressbar import ProgressBar
import nltk

import boilerpipe
import db
from settings import BLACKLIST
from utils import split_list

def is_downloaded(url):
    """Return if the url have already been downloaded or not.

    :param url: the url to check
    """
    exists = db.resources.one({'url': url, 'processed': True})
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
    content = ""

    # do not download if already downloaded or blacklisted
    if not is_blacklisted(url) and not is_downloaded(url):
        # 5s is agressive but we need to process tons of urls
        with contextlib.closing(urllib2.urlopen(url, timeout=5)) as file:
            # read the content and store it into a resource document
            content = file.read()
            charset = chardet.detect(content)
            if 'encoding' in charset and charset['encoding']:
                content = content.decode(charset['encoding'])
            else:
                content = ""

    return content

def reset():
    """Remove all the contents from the resources (+ unblacklist them).
    """
    progress = ProgressBar()
    resources = list(db.resources.Resource.find())
    if resources:
        for res in progress(resources):
            res.processed = False
            res.blacklisted = False
            res.content = None
            res.save()

def process_resources(threads):
    """Download all the unprocessed resources
    """
    class Downloader(Thread):
        def __init__(self, resources):
            self.resources = resources
            super(Downloader, self).__init__()

        def run(self):
            boilerpipe.jpype.attachThreadToJVM()
            for res in self.resources:
                try:
                    content = download(res.url)
                    content = boilerpipe.transform(content)
                except:
                    content = ""

                if content and len(content) >= 200:
                    res.textual = True

                # we don't want documents of less that 25 chars
                if not content:
                    res.blacklisted = True
                    print "blacklisted %s" % res.url
                else:
                    res.content = content
                    print "downloaded %s" % res.url
                res.processed = True
                res.save()

    if threads > multiprocessing.cpu_count():
        threads = multiprocessing.cpu_count()

    # initialise the JVM
    boilerpipe.start_jvm()
    resources = list(db.resources.Resource.find({'processed': False}))

    print "download %s urls using %s threads" % (len(resources), threads)
    
    # split the resource into the number of threads
    resources = split_list(resources, threads)

    # start the threads and pass them the resources to be processed
    for i in range(threads):
        d = Downloader(resources[i])
        d.start()


def main(r=False):
    if r:
        reset() 
    else:
        process_resources(int(sys.argv[1]) if len(sys.argv) > 1 else 1)


if __name__ == '__main__':
    main(r=len(sys.argv) > 1 and sys.argv[1] == "reset")
