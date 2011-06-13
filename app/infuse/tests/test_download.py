"""This file contains the tests for the download.py file."""

from unittest2 import TestCase
from infuse import download

# Those are mocks for the tests. The goal here is to replace the real
# implementation of the storage mechanism. That way, we can tests that it is 
# behaving the way we want to.
def fake_urlopen(*args, **kwargs):
    class FakeFile(object):
        def __init__(self):
            self.closed = False

        def read(self):
            return "this is the content"

        def close(self):
            self.closed = True

    return FakeFile()

class FakeResource(object):
    items = []

    class Resource(object):
        def __init__(self):
            self.url = None
            self.content = None

        def save(self):
            FakeResource.items.append(self)

    def one(*args, **kwargs):
        return None

fake_resource = FakeResource()

# Here come the actual test cases
class TestDownload(TestCase):

    def test_is_blacklisted(self):
        # first thing, monkey patch the list of blacklisted urls  
        self.old_blacklist = download.BLACKLIST
        download.BLACKLIST = ['blacklisted', 'black-listed']

        try:
            # *.example.* should be blacklisted, thus example.org, example.com etc.
            for url in ['http://fr.blacklisted.org', 
                    'http://blacklisted.org/content/', 
                    'http://black-listed.com/content/']:
                self.assertTrue(download.is_blacklisted(url))

            # default url shouldn't be blacklisted
            for url in ['http://example.org/content',]:
                self.assertFalse(download.is_blacklisted(url))

            # all protocols other than simple HTTP(S) should be blacklisted
            for url in ['ftp://example.org/test/test', 'sftp://example.org']:
                self.assertTrue(download.is_blacklisted(url))

        finally:
            # put back the original values for BLACKLIST
            download.BLACKLIST = self.old_blacklist

    def test_download(self):
        # monkeypatch urllib2.urlopen
        self._old_urlopen = download.urlopen
        download.urlopen = fake_urlopen

        # the blacklist
        self.old_blacklist = download.BLACKLIST
        download.BLACKLIST = ['blacklisted', 'black-listed']

        # test that the content is downloaded
        try:
            content = download.download("http://example.org/test")
            self.assertEqual(content, "this is the content")
        finally:
            # restore the monkey patched callables
            download.urlopen = self._old_urlopen
            download.BLACKLIST = self.old_blacklist
