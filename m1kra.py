# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:34:40 2017

@author: AKUPPAM
"""
""" 
Installed mapbox from https://pypi.python.org/pypi/mapbox/0.10.1#downloads
Dowloaded the gz/tar file
Ran 'pip install mapbox' from the Anaconda cmd prompt, and succefully installed mapbox

"""

from mapbox import Geocoder
geocoder = Geocoder()

import os
geocoder.session.params['access_token'] == os.environ['MAPBOX_ACCESS_TOKEN']

# --------------------
# conftest.py

import os

import pytest


@pytest.fixture
def uploads_dest_id():
    version = os.environ.get('TRAVIS_PYTHON_VERSION', 'test')
    return 'uploads-{0}'.format(version.replace(".", "-"))

# --------------------
# distance.py

from uritemplate import URITemplate

from mapbox.encoding import encode_coordinates_json
from mapbox.errors import InvalidProfileError
from mapbox.services.base import Service


class Distance(Service):
    """Access to the Distance API."""

    baseuri = 'https://api.mapbox.com/distances/v1/mapbox'
    valid_profiles = ['driving', 'cycling', 'walking']

    def _validate_profile(self, profile):
        if profile not in self.valid_profiles:
            raise InvalidProfileError(
                "{0} is not a valid profile".format(profile))
        return profile

    def distances(self, features, profile='driving'):
        profile = self._validate_profile(profile)
        coords = encode_coordinates_json(features)
        uri = URITemplate(self.baseuri + '/{profile}').expand(profile=profile)
        res = self.session.post(uri, data=coords,
                                headers={'Content-Type': 'application/json'})
        self.handle_http_error(res)
        return res
# ---------------------------------------
        
from mapbox import Distance

service = Distance()


# The input waypoints to the  directions  method are features, typically GeoJSON-like feature dictionaries.

portland = {
    'type': 'Feature',
    'properties': {'name': 'Portland, OR'},
    'geometry': {
        'type': 'Point',
        'coordinates': [-122.7282, 45.5801]}}
bend = {
    'type': 'Feature',
    'properties': {'name': 'Bend, OR'},
    'geometry': {
        'type': 'Point',
        'coordinates': [-121.3153, 44.0582]}}
corvallis = {
    'type': 'Feature',
    'properties': {'name': 'Corvallis, OR'},
    'geometry': {
        'type': 'Point',
        'coordinates': [-123.268, 44.5639]}}


# The  distance  method can be called with a list of point features and the travel profile.

response = service.distances([portland, bend, corvallis], 'driving')
response.status_code
200
response.headers['Content-Type']
'application/json; charset=utf-8'


# And the response JSON contains a durations matrix, a 2-D list with travel times (seconds) between all input waypoints. The diagonal will be zeros.

from pprint import pprint
pprint(response.json()['durations'])
[[0, ..., ...], [..., 0, ...], [..., ..., 0]]


# See  import mapbox; help(mapbox.Distance)  for more detailed usage.

# -------------------------

import SimpleHTTPServer
import SocketServer

PORT = 8000

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

httpd = SocketServer.TCPServer(("", PORT), Handler)

print ("serving at port", PORT)
httpd.serve_forever()

# -------------------------------------
import http.server

def start_server(port=8000, bind="", cgi=False):
    if cgi==True:
        http.server.test(HandlerClass=http.server.CGIHTTPRequestHandler, port=port, bind=bind)
    else:
        http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler,port=port,bind=bind)

start_server(cgi=True) #If you want cgi, set cgi to True e.g. start_server(cgi=True)

# --------------------------------------

import http.server
import socketserver

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

httpd = socketserver.TCPServer(("", PORT), Handler)

print("serving at port", PORT)
httpd.serve_forever()




