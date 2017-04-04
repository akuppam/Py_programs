# 'from' lines
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	if re.search('From:', line):
		print line
print "done 1"
print "================="
# 'from' lines - different way
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	if re.search('^From:', line):
		print line
print "done 2"
print "================="
# any no of times
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	if re.search('^X.*:', line):
		print line
print "done 3"
print "================="
# any no of times - certain kind
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	if re.search('^X-DSPAM-P\S+:', line):
		print line
print "done 4"
print "================="
# find numbers in the text
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	x = re.findall('[0-9]+', line)
	print x
print "done 5"
print "================="
# find numbers in the text
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	x = re.findall('[0-9].+', line)
	print x
print "done 5a"
print "================="
# find vowels in the text
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	x = re.findall('[AEIOU]+', line)
	print x
print "done 6"
print "================="
# extracting phrases between white spaces
import re
hand = open ('mbox-short.txt')
for line in hand:
#	line = line.rstrip()
	x = re.findall('\S+@\S+', line)
	print x
print "done 7"
print "================="
