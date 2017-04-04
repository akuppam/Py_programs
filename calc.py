#############
x=2
y=3*x
z=(x+10*y)/2
print z
#############
# write text out and enter a value to it
###########

x = raw_input ("what's your pay rate per hour: ")
pay = float(x)
y = raw_input("how many hours per week you work: ")
hours = float(y)
z = pay * hours
weekwage = raw_input("my weekly wages are: ")
print z
#############
# use decimal point and float to get accurate values
###########

x = raw_input ("enter C: ")
cen = float(x)
y = cen * (9.0/5.0) + 32
fahr = float(y)
z = raw_input("equivalent F: ")
print (fahr)
#############

if fahr<=50:
 print 'cold'
if fahr>=100:
 print 'hot'
if fahr>50:
    print 'warm'
#############
# defining and storing a function
#############
def xyz():
 print 'Hi'
 print 'Ronith'
 n = raw_input ("enter a number: ")
 num = float(n)
 print (num)
 
xyz()
################
big = max('hello ronith')
print big
tiny = min('hello ronith')
print tiny
#############
# using other functions like 'def', 'return'
#############

x = raw_input ("what's your pay rate per hour: ")
pay = float(x)
y = raw_input("how many hours per week you work: ")
hours = float(y)

def sal(pay,hours):
 if hours>=40:
  s = pay * hours
 else:
  s = pay * hours * 0.75
 print "Salary = $", s
 return s
sal(pay,hours)

####################
n = 0
while n>0:
 print 'xyz'
 print 'abc'
 
print 'lmn'
###################
2

