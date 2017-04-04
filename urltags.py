import urllib
from BeautifulSoup import *
url = raw_input('Enter - ')
html = urllib.urlopen(url) .read()
soup = BeautifulSoup(html)
tags = soup('a')
for tag in tags:
	print tag.get('href', None)
	
# this program will prompt for a website
# enter starting with "http", that is, http://www.camsys.com
# make sure you have 'beautifulsoup.py in the same folder as the python code that you are tryin to run
