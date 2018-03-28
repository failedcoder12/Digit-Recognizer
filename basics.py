
#Dictionary

dict = {}
dict['one'] = "This is the one"
dict[2] = "This is the two"

tinydict = {'name':'john','code':6743,123:'pokemin'}

print(dict['one'])
print(dict[2])
print(tinydict)
print(tinydict.keys())
print(tinydict.values())

#Basics

d,e,f = 1,2,"Pokemon"

print(d,f,e)

com = 10 + 5j

print(type(com))

#Lists

list = ['abcd', 786, 2.23, 'john', 70.2]
tintlist = [123,'john']
print(list[1:3])
print(list[2:])
print(tintlist*2)
print(list+tintlist)

#Bases
string1 = "123"
string2 = "11111"

num = int(string1)
print(num)

num1 = int(string2,2)
print(num1)

#Inputs and ifelse

temp = int(input("What is the temperature ? "));
if(temp>35):
	print("Too Hot today!")

import numpy as np

def quad_solver(a,b,c):

	dis = b*b - 4*a*c
	if(dis>=0):
		sqrt = np.sqrt(dis)
		root1 = (-b + sqrt)/(2*a)
		root2 = (-b - sqrt)/(2*a)
		print("Roots = ",root1,root2)
	else:
		print("No real roots")
	return

(a,b,c) = [int(x) for x in input("Input coefficient a,b,c :").split()]

quad_solver(a,b,c)

def max_3(x1,x2,x3):
	if(x1 > x2 and x1 > x3):
		mx = x1
	elif(x2 >= x1 and x2 >= x3):
		mx = x2
	else:
		mx = x3
	return mx

(x1,x2,x3) = [int(x) for x in input('Enter 3 numbers to compare').split()]

print(max_3(x1,x2,x3))

#For loops

for i in range(5):
	print(i)

for i in range(3,11,2):
	print(i)

for i in np.arange(1.5,5,0.5):
	print(i)

#While loop

count = 0

while(count < 9):
	print('THe count is: ',count)
	count = count + 1
else:
	print("Loop Over")

print("Good Bye")


#Functions
def add_interest(bal_list,rate):
	for i in range(len(bal_list)):
		bal_list[i] += (bal_list[i]*rate)
	return

amounts = [1000,2000,3000,4000,5000]
rate = 0.05

add_interest(amounts,rate)

print(amounts)

# Plot the following signals

#     x[n]=cos(πn4)x[n]=cos⁡(πn4)
#     x[n]=e(πn4)

import matplotlib.pyplot as pyplot

_x = np.arange(0,10,0.1)
_y = np.cos(-_x*np.pi/4)
print(_x)
print(_y)

pyplot.stem(_x,_y)
pyplot.show()

# Matrics

x1 = np.arange(3)
x2 = np.arange(3,6)
print(x1)
print(x2)
print(np.add(x1, x2))

a = np.matrix('1 2 3; 4 5 6; 7 8 9')
print(a)

b = np.matrix('10 20 30; 40 50 60; 70 80 90')
print(b)

print(np.add(a,b))

print(np.matmul(a,b))

print(np.power(a,3))

a = np.matrix('2 4 6; 4 8 6; 2 8 10')
b = 2

print(np.divide(a,b))

#Inverse of Matrix
ainv = np.linalg.inv(a)
print(ainv)

# Plotting
# -------------------------------
# ------------------------
# +++++++++++++++++
# +++++++++++
# -------
# ++++
# ==
# -

x = np.arange(np.pi,2*np.pi,np.pi/10)
y = np.cos(x)
z = np.sin(x)

#Sharing x axis
f,axarr = pyplot.subplots(2, sharex=True)
axarr[0].stem(x,y)
axarr[0].set_title('Cosine Plot')
axarr[1].stem(x,z)
axarr[1].set_title('Sine Plot')
pyplot.show()

x = np.arange(-1,1,0.1)
y = np.sign(x)
z = np.exp(x)

#Sharing y axis
f,(ax1,ax2) = pyplot.subplots(1,2,sharex=True)
ax1.stem(x,y)
ax1.set_title('Signum function')
ax2.stem(x,z)
ax2.set_title('Exponential function')

pyplot.show()

x = np.arange(0,2*np.pi,np.pi/15)
y1=2*np.cos(x)
y2=np.cos(x)
y3=0.5*np.cos(x)
fig,ax = pyplot.subplots()

line1 = ax.plot(x,y1,'--',linewidth=2,dashes=[20,5,20,5],label='2cos(x)')
line2 = ax.plot(x,y2,linestyle='--',color='r',linewidth=3,label='cos(x)')
line3 = ax.plot(x,y3,marker='v',linewidth=1,label='0.5cos(x)')
ax.legend(loc='lower right')
pyplot.show()

#LOG PLOT

t = np.arange(0.01,20.0,0.01)
pyplot.semilogy(t,np.exp(-t/5.0))
pyplot.title('Semilogy')
pyplot.grid(True)
pyplot.show()

pyplot.semilogy(t,np.sin(2*np.pi*t))
pyplot.title('Semilogx')
pyplot.grid(True)
pyplot.show()

#Histogram

mu,sigma = 2,0.5
v = np.random.normal(mu,sigma,10000)
pyplot.hist(v,bins=50,normed=1)
pyplot.show() 

#Image

import matplotlib.image as mpimg

img = mpimg.imread('image.png',0)
imgplot = pyplot.imshow(img)

pyplot.hist(img.ravel(),bins=256)
pyplot.show()