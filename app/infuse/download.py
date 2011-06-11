"""Download a web content if not already downloaded.

The results are stored in a temporary database "resources".
"""
from urlparse import urlparse
import urllib2
import contextlib
import re
import fnmatch

import pika
from pika.adapters import SelectConnection
import chardet

from db import resources
from settings import BLACKLIST
import boilerpipe

# init queue
connection = None
channel = None

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
        resource = resources.Resource.get_or_create(unicode(url))
        try:
            # 5s is agressive but we need to process tons of urls
            with contextlib.closing(urllib2.urlopen(url, timeout=5)) as file:
                # read the content and store it into a resource document
                content = file.read()
                charset = chardet.detect(content)
                if 'encoding' in charset and charset['encoding']:
                    content = content.decode(charset['encoding'])
                    resource.processed = True
                    resource.content = boilerpipe.transform(content)
                    print "saved %s" % resource.url
                else:
                    # skip this one
                    blacklist(resource)

        except urllib2.URLError:
            # TODO mark it as unusable, delete related views
            blacklist(resource)
        resource.save()

def blacklist(resource):
    print "blacklist %s" % resource.url
    resource.processed = True
    resource.blacklisted = True

def main():
    """Listen for events on the queue and download them"""

    global connection
    global channel

    def on_connected(connection):
        connection.channel(on_channel_open)

    def on_channel_open(channel_):
        global channel
        channel = channel_
        channel.queue_declare(queue="download_resource", durable=True,
                              exclusive=False, auto_delete=False,
                              callback=on_queue_declared)

    def on_queue_declared(frame):
        channel.basic_consume(handle_delivery, queue='download_resource')
        print ' [*] Waiting for messages. To exit press CTRL+C'

    def handle_delivery(channel, method_frame, header_frame, body):
        download(body)
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)

    parameters = pika.ConnectionParameters("localhost")
    connection = SelectConnection(parameters, on_connected)

    # initialise the JVM
    boilerpipe.start_jvm()

    try:
        connection.ioloop.start()
    except KeyboardInterrupt:
        connection.close()
        connection.ioloop.start()
        boilerpipe.stop_jvm()
 
if __name__ == '__main__':
    main()
