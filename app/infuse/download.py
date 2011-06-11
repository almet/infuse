"""Download a web content if not already downloaded.

The results are stored in a temporary database "resources".
"""
from urlparse import urlparse
import urllib2
import contextlib
import re
import fnmatch

import pika
import chardet

from db import resources
from settings import BLACKLIST

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
        try:
            # 5s is agressive but we need to process tons of urls
            with contextlib.closing(urllib2.urlopen(url, timeout=5)) as file:
                # read the content and store it into a resource document
                resource = resources.Resource.get_or_create(unicode(url))
                content = file.read()
                charset = chardet.detect(content)
                if 'encoding' in charset and charset['encoding']:
                    content = content.decode(charset['encoding'])
                    resource.processed = True
                    resource.save()
                    print "saved %s" % resource.url
                else:
                    # skip this one
                    blacklist_url(url)

        except urllib2.URLError:
            # TODO mark it as unusable, delete related views
            blacklist_url(url)

def blacklist_url(url):
    print "blacklist %s" % url

def main():
    """Listen for events on the queue and download them"""

    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='download_resource', durable=True)

    print ' [*] Waiting for messages. To exit press CTRL+C'
    def callback(ch, method, properties, body):
        download(body)

    channel.basic_consume(callback, queue='download_resource', no_ack=True)
    channel.start_consuming()
 
if __name__ == '__main__':
    main()
