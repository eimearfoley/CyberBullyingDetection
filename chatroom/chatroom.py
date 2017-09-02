#!/usr/bin/env python3

from cgitb import enable
enable()

from os import environ
from cgi import FieldStorage, escape
from time import time
import os
import threading

os.environ['http_proxy']="http://4c.ucc.ie:80"
os.environ['https_proxy']="http://4c.ucc.ie:80"

form_data = FieldStorage()
message=''

if len(form_data)!=0:
    message = escape(form_data.getfirst('message', '').strip())
    fh = open("/var/www/html/log.txt", "a")
    fh.write("\n"+message)
    fh.close()

print('Content-Type: text/html')
print()
print(""" <!DOCTYPE html>
        <html lang="en">
                <head><title>Login</title>
                <script src="../chatroom.js"></script>
                </head>
                <body>
                        <h1>Chatroom</h1>
                <table class="chatroom">
                <tr><td><span id="display"></span></td></tr>

                <form action="chatroom.py" method="get">
                <tr><td><label for="message">Message:</label>
                <input type="text" name="message" id="message"></td></tr>
                <tr><td><span id="click"></span></td></tr>
                </form>""")
