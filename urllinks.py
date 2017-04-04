# Note - this code must run in Python 2.x and you must download
# http://www.pythonlearn.com/code/BeautifulSoup.py
# Into the same folder as this program

import urllib
from BeautifulSoup import *

url = raw_input('Enter - ')
html = urllib.urlopen(url).read()
soup = BeautifulSoup(html)

# Retrieve all of the anchor tags
tags = soup('a')
for tag in tags:
    print tag.get('href', None)

#########################
	
import urllib

link = "https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Muran.html"
f = urllib.urlopen(link)
myfile = f.read()
print myfile

#########################

from bs4 import BeautifulSoup
import urllib2

resp = urllib2.urlopen("https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Muran.html")
soup = BeautifulSoup(resp, from_encoding=resp.info().getparam('charset'))

for link in soup.find_all('a', href=True):
    print link['href']
