# find numbers in the text
# this lists all the number in various 'lists[]'
# this also indicates how many columns of numbers exist in each list
# for example, ['12','3'], has two columns of data
# each column of data needs to be added separately and then the column sums have to be added
import re
hand = open ('mbox-short.txt')
for line in hand:
	line = line.rstrip()
	x = re.findall('^New Revision:.*[0-9]+', line)
	print x
print "done listing New Revision numbers in data"
print "================="
# find numbers in the text
import re									# importing regular expression (re) library)
hand = open ('mbox-short.txt')			# open and read the *.txt file
intlist = list()							# initialize an empty list to store all numbers

for line in hand:							# starting a for loop
	line = line.rstrip()					# removing any white space - non-essential step
	x = re.findall('^New Revision:.* ([0-9]+)', line)			# find any numbers ranging from 0 to 9 in all lines of the text file
	print x
	if len(x) > 0:							# if length of the list is equal to 1, that is, if there is only 1 number in quotes (e.g., ['12'])
		integers1 = int(x[0])				# initialize 'integers1' and make it equal to the 1st number (or 1st column of numbers) in those lines that has only 1 number in quotes
		intlist.append(integers1)			# append all numbers or integers from the 1st column of integers into "intlist"
sumnum = float(0)									# initialize a new variable
count = float(0)
for number in intlist:						# for loop to identify every integer in the intlist
	sumnum += number						# adding all integers and storing in the new variable
	count = count + 1
print sumnum
print count
average = (sumnum / count)
print 'Average:', average

print "done adding all columns of numbers and averaging them"
print "================="
#Optional: Just for Fun 
#There are a number of different ways to approach this problem. While we don't recommend trying to write the most compact code possible, it can sometimes be a fun exercise. Here is a a redacted version of two-line version of this program using list comprehension: 
#import re
#print sum( [ ****** *** * in **********('[0-9]+',**************************.read()) ] )

#Please don't waste a lot of time trying to figure out the shortest solution until you have completed the homework. List comprehension is mentioned in Chapter 10 and the read() method is covered in Chapter 7. 