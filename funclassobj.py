# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:22:30 2016

@author: AKUPPAM
"""
def pqr(x):
    y = x - 3*x + 4/x + 10
    return y
 
pqr(2)
############################
def pqrs():
    print ('ha ha ha ')

pqrs()
############################
def print_lyrics(): 
    print ("I'm a lumberjack, and I'm okay.")
    print ('I sleep all night and I work all day.')

print_lyrics()
############################
def computepay(hours, rate):
    pay = ((hours - 40) * (rate + (0.5*rate)) + (40 * rate))
    return pay

computepay(45, 10)

###########################
""" Exercise 4.7 Rewrite the grade program from the previous chapter using a func- tion 
called computegrade that takes a score as its parameter and returns a grade as a string. 

Score Grade > 
0.9 A > 
0.8 B > 
0.7 C > 
0.6 D 
<= 0.6 F 

Program Execution: Enter score: 0.95  """

def computegrade(score):
    if (score <= 1.0 and score > 0.9):
        return 'A'
    elif 0.8 <= score < 0.9:
        return 'B'
    elif 0.7 <= score < 0.8:
        return 'C'
    elif 0.6 <= score < 0.7:
        return 'D'
    elif 0.0 <= score < 0.6:
        return 'F'
    else:
        return 'Bad score'
            
computegrade(0.45)
computegrade(0.95)
computegrade(0.65)
computegrade(0.75)
computegrade(1.75)
""" Note: (1) with or without parantheses after 'if' works fine (2) 'and' works but not '&'
(3) either use 'and' or put the parameter in the middle (4) everything outside range is 'bad score' 
"""
###########################
def funcset():
    score = 0.386    
    print(computegrade(score))
    print(computegrade(0.775))
    print_lyrics()
    hours = 100
    rate = 200    
    print(computepay(hours, rate))
    print(computepay(120, 200))
    x = 1
    print(pqr(x))
    pqrs()

funcset()
###########################
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)
##################################
"""
USE THIS LINK FOR "CLASSES" https://docs.python.org/3/tutorial/classes.html
AFTER THIS, MOVE ON TO MODULES
AND THEN GO THROUGH MAG AND MARC PY PROGRAMS
IF STILL NOT CLEAR, SEARCH FOR EXAMPLES OF FUNCTIONS AND CLASSES
"""
class aclass():
    i = 12345
    
    def rnfunc():
        print ('who hoo')
        
aclass.i
aclass.rnfunc()
########################
class Complex:
    def __init__(self, realpart, imagpart):
        self._r = realpart
        self._i = imagpart

x = Complex(3.0, -4.5)
x._r, x._i

class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

x = Complex(3.0, -4.5)
x.r, x.i

class Complex:
    def __init__(self, realpart, imagpart):
        self.rrr = realpart
        self.iii = imagpart

x = Complex(3.0, -4.5)
x.rrr, x.iii

##############################

class Complex:
    def __init__(self, realpart, imagpart):
        self._r = realpart
        self._i = imagpart
    def writevalues(self):
        print (self._r)
        print (self._i)
        print (self._r, self._i)

x = Complex(3.0, -4.5)
x._r, x._i
Complex.writevalues(x)
##############################
"""
(1) first, define a name of the class (start with an UPPER case letter)
(2) second, define one or more functions inside the class
(3) third, create an 'instance' using the class name (e.g., x = Message('la la la))
(4) fourth, execute one or more functions inside the class by calling Class.Function(instance)
(5) fifith, (optional), a second function can also be executed by calling Class.Function(instance); 
(note - instance does not always have to have a value)
""" 

class Message:
    def __init__(self, aString):
        self.text = aString
    def printIt(self):
        print (self.text)
    def printAlso():
        print ('class was called sucessfully')
        
x = Message('la la la')
Message.printIt(x)
Message.printAlso()

#############################

class C:
    def __init__(self, val):
        self.val = val
    def printval(self):
        print ('hello, my value is:', self.val)

val = C(27)
C.printval(val)

#############################

class ABC:
    def __init__(self, vals):
        self.vals = vals
    def printvals(self):
        print ('My values are: ', self.vals)

vals = ABC(107.45)
ABC.printvals(vals)

"""
define a class
include two functions inside it
one, that prints a text
second, that takes in an instance value
"""

class Person1:
    def __init__(self, weight):
        self.wt = weight
    def printwt(self):
        print ('Person1 weighs', self.wt, 'lbs')
        
wt = Person1(20)
Person1.printwt(wt)

class Factory:
    def __init__(self, tons):
        self.tons = tons
    def printtons(self):
        print ('This Factory produces', self.tons, 'lbs on a annual basis')
        
tons = Factory(4500)
Factory.printtons(tons)

class Shipper:
    def __init__(self, value_of_goods):
        self.goods = value_of_goods
    def printgoodsvalue(self):
        print ('The goods shipped by Shipper is valued at $', self.goods, 'in 2016 dollars')

goodsval = Shipper(55675.34)
Shipper.printgoodsvalue(goodsval)

class Shipper:
    def __init__(self, value_of_goods):
        self.goods = value_of_goods
        print ('The goods shipped by Shipper is valued at $', self.goods, 'in 2016 dollars')

goodsval = Shipper(55675.34)

#####################################

class Shipper2:
    def __init__(self, value_of_goods_usd):
        self.goodsval = value_of_goods_usd
    def printgoodsval(self):
        print ('Value of Shipper2 goods is', self.goodsval, 'in British pounds.')
        return '$', self.goodsval/0.77, 'USD'
        
goodsval = Shipper2(67895.25)
Shipper2.printgoodsval(goodsval)

# instance values list
valuelist = [Shipper2(20000), Shipper2(40000), Shipper2(60000), Shipper2 (80000)]

for goodsval in valuelist:
    print ('Value of Shipper2 goods is', Shipper2.printgoodsval(goodsval))
    
###################################
    
# "Exception" is a default class in Python - hence shows up in Purple color
class BalanceError(Exception):
    value = "no balance"
    
class BankAccount:
    def __init__(self, initialAmount):
        self.inAmt = initialAmount
        print ("your balance is $%8.2f" %self.inAmt)
        
    def deposit(self, amount):
        self.balance = self.inAmt + amount
        print ("your new balance after depositing is $%8.2f" %self.balance)
            
    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance = self.balance - amount
        else:
            raise BalanceError
        print ("your new balance after withdrawing is $%8.2f" %self.balance)
                
""" different ways to get account balance """
balance = BankAccount(25)
inAmt = BankAccount(35)
amount = BankAccount(45)

""" ideal way to test the whole set of code: balance, deposit, withdraw """
""" the code keeps deductinng money as you keep executing withdraw code """
a = BankAccount(500)
a.deposit(600)
a.withdraw(10)


