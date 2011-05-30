"""This file contains the convertion logic from raw data to something which actually
means something valuable.

The raw data only contains data about the events that have been fired-up on the web
browser, so we do have informations like "this tab have been opened", "this tab have
been closed" and so on. This module convert this kind of information into 
information like "the user have watched this resource for XX seconds", 
"he watched it N times" etc.

The implementation is based on Map/Reduce functions that are directly executed on
the mongodb server.
"""
