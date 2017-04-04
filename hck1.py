# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:20:36 2016

@author: AKUPPAM
"""

""" [Go to >> Consoles >> Open a Python Console] to see each code of line runing/outputting """

""" Hackerrank.com 
Apple and Orange """

s = 7
t = 11
a = 5
b = 15
m = 3
n = 2

AppleFall_1 = -2
AppleFall_2 = 2
AppleFall_3 = 1

OrangeFall_1 = 5
OrangeFall_2 = -6

AppleFall_1_fromTree = a + AppleFall_1
AppleFall_2_fromTree = a + AppleFall_2
AppleFall_3_fromTree = a + AppleFall_3

AppleFall_1_fromTree
AppleFall_2_fromTree
AppleFall_3_fromTree

OrangeFall_1_fromTree = b + OrangeFall_1
OrangeFall_2_fromTree = b + OrangeFall_2

if 7 <= AppleFall_1_fromTree <= 11:
    apple1 = 1
else:
    apple1 = 0

if 7 <= AppleFall_2_fromTree <= 11:
    apple2 = 1
else:
    apple2 = 0

if 7 <= AppleFall_3_fromTree <= 11:
    apple3 = 1
else:
    apple3 = 0

apples = apple1 + apple2 + apple3
#print ("No of Apples falling within the region:", apples)


if 7 <= OrangeFall_1_fromTree <= 11:
    orange1 = 1
else:
    orange1 = 0

if 7 <= OrangeFall_2_fromTree <= 11:
    orange2 = 1
else:
    orange2 = 0

oranges = orange1 + orange2

apples
oranges
#print ("No of Oranges falling within the region:", oranges)

x = input("enter C: ")
cen = float(x)
y = cen * (9.0/5.0) + 32
fahr = float(y)
z = input("equivalent F: ")
print (fahr)

# ----------------------------------------

n=input("enter the board size ")
print (" ---"*n)
n=3
for r in range(1,n+1):
 print ("| "*(n+1))
 print (" ---"*n)

# ---------------------

for i in range(1, 8):
    if i >= 5:
        break
    # do something
        
        #!/bin/python
"""
import sys
N = int(raw_input().strip())
"""

i = input()

i = 28
if (i % 2 == 0):
    print ("")
else: print ('Weird')

if i in range(2,5):
    if (i % 2 == 0):
        print ("Not Weird")

if i in range(6,20):
    if (i % 2 == 0):
        print ("Weird")

if i > 20:
    if (i % 2 == 0):
        print ("Weird")

""" Task 
 Given an integer, , perform the following conditional actions:
•If n is odd, print Weird
•If n is even and in the inclusive range of 2 to 5, print Not Weird
•If n is even and in the inclusive range of 6 to 20, print Weird
•If n is even and greater than 20, print Not Weird
"""
# Python 2 

n = int(raw_input())
if n % 2 == 1:
    print "Weird"
elif n % 2 == 0 and 2 <= n <= 5:
    print "Not Weird"
elif n % 2 == 0 and 6 <= n <= 20:
    print "Weird"
else:
    print "Not Weird"


# Python 3 

n = int(input())
if n % 2 == 1:
    print("Weird")
elif n % 2 == 0 and 2 <= n <= 5:
    print("Not Weird")
elif n % 2 == 0 and 6 <= n <= 20:
    print("Weird")
else:
    print("Not Weird")


