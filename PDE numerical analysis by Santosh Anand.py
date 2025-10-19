#python program to solve Poisson's equation......................................................................... 
import numpy as np
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
i=int(x_b/h)
j=int(t_b/h)
def f(x,y):
    return 10*(x**2 + y**2 + 10)*(h**2)
u=np.zeros(shape=(i+1,j+1))
u_temp=np.zeros(shape=(i+1,j+1))
# BOUNDARY CONDITIONS
u[:,0]=0
u[:,i]=0
#INITIAL CONDITIONS
u[0,:]=0
u[j,:]=0
u_temp=u.copy()
for it in range(50):
    u[:]=u_temp[:]
    for row in range(1, j):
        for col in range(1, i):
            u_temp[row, col] = 0.25 * (u[row+1, col] + u[row-1, col] + u[row, col+1] + u[row, col-1] +f((j-row)*h, (col*h)))
    diff=np.abs(u-u_temp)
    if np.all(diff<10**(-6)):
        print(it)
        break
print(u)


#python program to implement Bender Schmidt formula.......................................
import numpy as np
a=float(input("enter the value of a="))
k=float(input("enter the value of b="))
h=float(input("enter the value of h="))
x_b=int(input("enter the boundary value of x="))
t_b=int(input("enter the boundary value of t="))
y=k/(a*h**2)
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+1))
u[:,0]=0
u[:,i]=8
for col in range(1,i):
    u[j,col]=col/4*(8-col/2)
for row in range(j-1,-1,-1):
    for col in range(1,i):
        u[row,col]=0.5*(u[row+1,col-1] + u[row+1,col+1])
print(u)


#python program to implement explicit formula to solve parabolic equation.......................................
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
y=k/(a*h**2)
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+1))
u[:,0]=0
u[:,i]=0
for col in range(1,i):
    u[j,col]=m.sin(3.14*col/3)
for row in range(j-1,-1,-1):
    for col in range(1,i):
        u[row,col]=y*u[row+1,col+1] + (1-2*y)*u[row+1,col] + y*u[row+1,col-1]
print(u)

#python program to solve hyperbolic equation using finite difference method................................................
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
def g(x):
    return 0
def f(x):
    return x*(4-x)
y=(k*a)/h
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+1))
#........................................................................<<<BOUNDARY CONDITIONS>>>......................................................................
u[:,0]=0
u[:,i]=0
#............................................................................<<< INITIAL CONDITIONS >>>............................................................................
for col in range(1,i):
    u[j,col]=f(col*h)
for col in range(1,i):
    u[j-1,col]=0.5*(u[j,col-1] + u[j,col+1]) + g(col*h)
for row in range(j-2,-1,-1):
    for col in range(1,i):
        if y==1:
            u[row,col]=u[row+1,col+1] - u[row+2,col] + u[row+1,col-1]
        else:
           u[row,col]=(y**2)*u[row+1,col-1]+2*(1-y**2)*u[row+1,col]+(y**2)*u[row+1,col+1]-u[row+2,col]
print(u)


#python program for Crank Nicolson method :)-..............................................................
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
y=k/(a*h**2)
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+1))
u_temp=np.zeros(shape=(j+1,i+1))
#      .......................................................<<<BOUNDARY CONDITIONS>>>..............................................
u[:,0]=0
temp=j*k
for row in range(0,j+1):
    u[row,i]=temp
    temp=temp-k
u_temp=u.copy()
for it in range(50):
    u[:]=u_temp[:]
    for row in range(0,j):
        for col in range(1,i):
            if y==1:
                u_temp[row,col]=0.25*(u[row+1,col-1] + u[row,col-1] + u[row+1,col+1] + u[row,col+1])
            else:
                u_temp[row,col]=(1/(2*(1+y)))*(y*u[row,col-1] + y*u[row,col+1] + y*u[row+1,col-1] +y*u[row+1,col+1]+2*(1-y)*u[row+1,col])
print(u)


#example 2.....................................................................
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
y=k/(a*h**2)
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+1))
u_temp=np.zeros(shape=(j+1,i+1))
#               .......................................................<<<BOUNDARY CONDITIONS >>>.........................................................
u[:,0]=0
u[:,i]=0
#              .............................................................<<<INITIAL CONDITIONS>>>................................................................
temp=h
for col in range(1,i):
    u[j,col]=100*temp*(1-temp)
    temp=temp+h
u_temp=u.copy()
for it in range(50):
    u[:]=u_temp[:]
    for row in range(0,j):
        for col in range(1,i):
            if y==1:
                u_temp[row,col]=0.25*(u[row+1,col-1] + u[row,col-1] + u[row+1,col+1] + u[row,col+1])
            else:
                u_temp[row,col]=(1/(2*(1+y)))*(y*u[row,col-1] + y*u[row,col+1] + y*u[row+1,col-1] +y*u[row+1,col+1]+2*(1-y)*u[row+1,col])
print(u)


#python program to implement Lax-Wendroff's method to solve first order hyperbolic equations :).........................................
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
def g(x):
    return x
y=k/h
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+2))
#........................................................................<<<BOUNDARY CONDITIONS>>>......................................................................
for alpha in range(0,i+1):
    u[0,alpha]=g(alpha*h)
for alpha in range(0,j+1):
    u[alpha,i+1]=((i+1)*h)-(alpha*0.125)
#............................................................................<<< INITIAL CONDITIONS >>>............................................................................
for beta in range(0,j+1):
    u[beta,0]=g(beta*k)
for row in range(1,j+1,):
    for col in range(1,i+1):
        u[row,col]=u[row-1,col] - (a*y/2)*(u[row-1,col+1] - u[row-1,col-1])
print(u)


#python program to implement Wendroff's method to solve first order hyperbolic equations:)....................................................
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
def g(x):
    return x
y=k/h
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+1))
#........................................................................<<<BOUNDARY CONDITIONS>>>......................................................................
for alpha in range(0,i+1):
    u[0,alpha]=g(alpha*h)
#............................................................................<<< INITIAL CONDITIONS >>>............................................................................
for beta in range(0,j+1):
    u[beta,0]=g(beta*k)
for row in range(1,j+1,):
    for col in range(1,i+1):
        u[row,col]=((1-a*y)/(1+a*y))*u[row-1,col] - ((1-a*y)/(1+a*y))*u[row,col-1] +u[row-1,col-1]
print(u)




        
        
       

