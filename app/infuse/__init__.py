import time

import convert, download, profiles

def run_all(reset=False):
    """Run all the pipeline:
        - convert the events into views
        - download the resources and extract them using boilerpipe
        - cluster the resources and get the profiles

    :param reset: reset the status of the database before launching the pipeline
    """
    if reset:
        convert.reset()
        download.reset()

    t0 = time.time()
    convert.extract_views()
    download.process_resources()
    profiles.find_profiles_tfidf()
    print 'processed all the steps in %s' % (time.time() - t0)

if __name__ == '__main__':
    run_all(True)
