import urllib
import xml.etree.ElementTree as ET

url = 'http://python-data.dr-chuck.net/comments_171772.xml'

while True:
    address = raw_input('Enter location: ')
    if len(address) < 1 : break

#    url = serviceurl + urllib.urlencode({'sensor':'false', 'address': address})
    print 'Retrieving', url
    uh = urllib.urlopen(url)
    data = uh.read()
    print 'Retrieved',len(data),'characters'
    print data
    tree = ET.fromstring(data)
    print 'Name:',tree.find('name').text 
    print 'Count:',tree.find('count').text
