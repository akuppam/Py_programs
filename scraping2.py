import urllib
from bs4 import BeautifulSoup
from BeautifulSoup import *
url = raw_input('Enter - ')
html = urllib.urlopen(url) .read()
soup = BeautifulSoup(html)

#table = soup.find('table', attrs={'class': 'comments'})
#print table.prettify()

#print soup.find_all("span", {"class" : "comments"})
#print(soup.get_text())

#Numbers = soup.find_all("span", {"class" : "comments"}).get_text()
#print Numbers
#print Numbers[0].text

#tags = soup('span class="comments"')
#tags = soup('span')
#tags = soup("span", {"class" : "comments"})
tags = soup("comments")
print tags
#print soup.find_all("span", {"class" : "comments"})

#for tag in tags:
#	print 'Name:',name
#	print 'URL:',tag.get('href', None)    
#	print 'Contents:',tag.contents[0]    
#	print 'Attrs:',tag.attrs
#	print 'Name:', tag.get('span class')
#	counts = soup.findall('.//span class="comments"')
#	print "counts[0] = ", counts[0]
#	print "counts[0].text = ", counts[0].text
#	print "type(int(counts[0].text)) = ", type(int(counts[0].text))
#	total = 0
#	for i in counts:
#		total = total + int(counts[i].text)
#	print total
#	break

	

#value = [item.get_text() for item in soup.find_all("td", {"name": "comments"})]

# <tr><td>Chelsi</td><td><span class="comments">98</span></td></tr>

  
# see this link for some clues:
# http://stackoverflow.com/questions/30810934/python-beautiful-soup-web-scraping-specific-numbers

	
# this program will prompt for a website
# enter starting with "http", that is, http://www.camsys.com
# make sure you have 'beautifulsoup.py in the same folder as the python code that you are trying to run
# it will print out all relevant text without the HTML garbage

