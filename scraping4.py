import urllib
from bs4 import BeautifulSoup
from BeautifulSoup import *
url = raw_input('Enter - ')
html = urllib.urlopen(url) .read()
soup = BeautifulSoup(html)

for tag in tags:
	print 'Name:',name
	print 'URL:',tag.get('href', None)    
	print 'Contents:',tag.contents[0]    
	print 'Attrs:',tag.attrs
	print 'Name:', tag.get('span class')
	counts = soup.findall('.//span class="comments"')
	print "counts[0] = ", counts[0]
	print "counts[0].text = ", counts[0].text
	print "type(int(counts[0].text)) = ", type(int(counts[0].text))
	total = 0
	for i in counts:
		total = total + int(counts[i].text)
	print 'Sum', total
	break
