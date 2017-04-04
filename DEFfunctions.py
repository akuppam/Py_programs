# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:13:04 2016

@author: AKUPPAM
"""

'''Function definition and invocation.'''

def happyBirthdayEmily():
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print("Happy Birthday, dear Emily.")
    print("Happy Birthday to you!")

happyBirthdayEmily()
happyBirthdayEmily()
happyBirthdayEmily()

# -----------------
def my_fx():
    print ('my_fx')
    
my_fx()
# -------------------------
def my_fx():
    print ("my function")
    
my_fx()
# --------------------------
def fx1(origin, dest):
    print("origin, %s, destination, %s")

fx1('dal', 'hou')
# -------------------------
def fx2(origin2, dest2):
    return origin2, dest2

fx2('dal', 'hou')

x = fx2('dal', 'hou')

print (x)

# --------------------------

# Modify this function to return a list of strings as defined above
def list_benefits():
    pass

# Modify this function to concatenate to each benefit - " is a benefit of functions!"
def build_sentence(benefit):
    pass

def name_the_benefits_of_functions():
    list_of_benefits = list_benefits()
    for benefit in list_of_benefits:
        print(build_sentence(benefit))

name_the_benefits_of_functions()

# ----------------------------------

class MyClass():
    variable = "blah"
    
    def function(self):
        print ("my fx")
        
myobject = MyClass()

myobject.variable

print (myobject.variable)

# -------------------------------
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()
myobjecty = MyClass()

myobjecty.variable = "yackity"

# Then pring out both values
print(myobjectx.variable)
print(myobjecty.variable)
# ---------------------------------

class vehicle:
    name = "jaguar"
    kind = "compact"
    color = "red"
    value = 1000.00
    def description(self):
        desc_veh = "%s is a %s %s worth $%.2f." %(self.name, self.kind, self.color, self.value)
        return desc_veh

car1 = vehicle()
car1.name = "honda"
car1.kind = "crv"
car1.color = "white"
car1.value = 2000.00

print(car1.description())

# -----------------------------

class Vehicle():
    name=""
    value=0.00
    def descrp(self):
        dveh = "%s is worth $%.2f." %(self.name, self.value)
        return dveh
    
car2 = Vehicle()
car2.name = "toyota"
car2.value = 1000.00

print(car2.descrp())
   
# ----------------------------

class comm():
    name=""
    tons=0
    
    def cg(self):
        cg="%s is more than %.2f." %(self.name, self.tons)
        return cg
    
cg1 = comm()
cg1.name="lumber"
cg1.tons=2000

print(cg1.cg())

# -----------------------------

class veh():
    name=""
    value=0.00
    def descr(self):
        descr = "%s is about $%.2f." % (self.name, self.value)
        return descr
    
truck1 = veh()
truck1.name="Tacoma"
truck1.value=20000

print(truck1.descr())

# ----------------------------------

class house():
    street=""
    price=0.00
    def home(self):
        home="%s is valued at $%.2f." %(self.street, self.price)
        return home
    
house1 = house()
house1.street="Capella"
house1.price=450000

print(house1.home())

# ------------------------------------------

# Dictionaries

phonebook = {}
phonebook["ax"]=767
print(phonebook)
# ----------------------------
value = {}
value["pr"]=45
print(value)

# ------------------------

bank = {}
bank['ac1']=25
bank['ac2']=45
print(bank)

# -------------------------------

class house():
    street=""
    value=0.00
    def descr(self):
        descr = "%s is priced at $%.2f." %(self.street, self.value)
        return descr
    
house1 = house()
house1.street = "Lane"
house1.value = 20000

print (house1.descr())

# ---------------------------------

bank = {}
bank["ac1"] = 25
bank["ac3"] = 35
print (bank)

# -----------------------------------

















