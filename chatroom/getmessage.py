#!/usr/bin/env python3
from cgitb import enable
enable()

from os import environ
import os

print("Content-Type: text/plain")
print()

os.environ['http_proxy']="http://4c.ucc.ie:80"
os.environ['https_proxy']="http://4c.ucc.ie:80"

file = open("/var/www/html/log.txt", "r")
entry = file.readlines()[-1]
print(entry)
file.close()
