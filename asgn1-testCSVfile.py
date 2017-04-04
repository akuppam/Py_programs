# testing asgn1.py on a different (csv) file (the total should be about 2.7427 x 10^13)
# find numbers in the text
# this lists all the number in various 'lists[]'
# this also indicates how many columns of numbers exist in each list
# for example, ['12','3'], has two columns of data
# each column of data needs to be added separately and then the column sums have to be added
import re
hand = open ('outcome-of-care-measures.csv')
for line in hand:
	line = line.rstrip()
	x = re.findall('[0-9]+', line)
	print x
print "done listing numbers in data"
print "================="
# find numbers in the text
import re									# importing regular expression (re) library)
hand = open ('outcome-of-care-measures.csv')			# open and read the *.csv file
intlist = list()							# initialize an empty list to store all numbers

for line in hand:							# starting a for loop
	line = line.rstrip()					# removing any white space - non-essential step
	x = re.findall('[0-9]+', line)			# find any numbers ranging from 0 to 9 in all lines of the text file
	if len(x) == 1:							# if length of the list is equal to 1, that is, if there is only 1 number in quotes (e.g., ['12'])
		integers1 = int(x[0])				# initialize 'integers1' and make it equal to the 1st number (or 1st column of numbers) in those lines that has only 1 number in quotes
		intlist.append(integers1)			# append all numbers or integers from the 1st column of integers into "intlist"
	if len(x) == 2:							# if length of the list is equal to 2, that is, if there are 2 numbers in quotes (e.g., ['12', '3'])
		integers2 = int(x[0])				# initialize 'integers2' and make it equal to the 1st number (or 1st column of numbers) in those lines that have 2 numbers in quotes
		integers3 = int(x[1])				# initialize 'integers3' and make it equal to the 2nd number (or 2nd column of numbers) in those lines that have 2 numbers in quotes
		intlist.append(integers2)			# append all numbers or integers from the 1st and 2nd column of integers into "intlist"
		intlist.append(integers3)
	if len(x) == 3:
		integers4 = int(x[0])
		integers5 = int(x[1])
		integers6 = int(x[2])
		intlist.append(integers4)
		intlist.append(integers5)
		intlist.append(integers6)
sumnum = 0									# initialize a new variable
for number in intlist:						# for loop to identify every integer in the intlist
	sumnum += number						# adding all integers and storing in the new variable
print sumnum

print "done adding all columns of numbers in data"
print "================="
#Optional: Just for Fun 
#There are a number of different ways to approach this problem. While we don't recommend trying to write the most compact code possible, it can sometimes be a fun exercise. Here is a a redacted version of two-line version of this program using list comprehension: 
#import re
#print sum( [ ****** *** * in **********('[0-9]+',**************************.read()) ] )

#Please don't waste a lot of time trying to figure out the shortest solution until you have completed the homework. List comprehension is mentioned in Chapter 10 and the read() method is covered in Chapter 7. 