#! /bin/bash

print "dumping the data on the server"
ssh notmyidea.org -p 20002 mongodump
print "downloading the dump from the server"
scp -rP 20002 notmyidea.org:/home/alexis/dump temp/mongosave
print "restore the data on the local mongodb instance"
mongorestore temp/mongosave
