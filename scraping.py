#import pickle
#import math
#import urllib2
#from lxml import etree
#from bs4 import BeautifulSoup
#from urllib import urlopen

import urllib
from BeautifulSoup import *
url = raw_input('Enter - ')
html = urllib.urlopen(url) .read()
soup = BeautifulSoup(html)

#tags = soup('a')
#for tag in tags:

for link in soup.find_all('a'):
    print(link.get('href'))
# http://example.com/elsie
# http://example.com/lacie
# http://example.com/tillie

#Another common task is extracting all the text from a page:

print soup.get_text()
# The Dormouse's story
#
# The Dormouse's story
#
# Once upon a time there were three little sisters; and their names were
# Elsie,
# Lacie and
# Tillie;
# and they lived at the bottom of a well.
#
# ...



#favPrevGMInfoUrl = 'http://www.cbssports.com/nfl/gametracker/boxscore/NFL_20140914_NE@MIN'
#favPrevGMInfoHtml = urlopen(favPrevGMInfoUrl).read()
#favPrevGMInfoSoup = BeautifulSoup(favPrevGMInfoHtml)
#favPrevGMInfo = favPrevGMInfoSoup.find_all("td", { "id" : "away-safeties" }).text

#print favPrevGMInfo
