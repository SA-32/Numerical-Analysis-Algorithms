'''def runge_kutta(func,derivfunc,a,b,n,y0,z0):
    h=(b-a)/n
    def df(x,y,z):
        df=eval(func)
        return df
    def df1(x,y,z):
        df1=eval(derivfunc)
        return df1
    for i in range(n):
         x_a=a + i*h
         k1=h*(df(x_a,y0,z0))
         l1=h*(df1(x_a,y0,z0))
         k2=h*(df(x_a+(h/2),y0+(k1/2),z0+(l1/2)))
         l2=h*(df1(x_a+(h/2),y0+(k1/2),z0+(l1/2)))
         k3=h*(df(x_a+(h/2),y0+(k2/2),z0+(l2/2)))
         l3=h*(df1(x_a+(h/2),y0+(k2/2),z0+(l2/2)))
         k4=h*(df(x_a+h,y0+k3,z0+l3))
         l4=h*(df1(x_a+h,y0+k3,z0+l3))
         y1=y0+(1/6)*(k1+2*k2+2*k3+k4)
         z1=z0+(1/6)*(l1+2*l2+2*l3+l4)
         y0=y1
         z0=z1
    return y1


# python program for secant method
def secant(y1,y2,f1,f2):
        yi=(y1*f2-y2*f1)/(f2-f1)
        return yi

    
#python program to implement shooting method

#value at x=1
target=2

#initial guesse of z at x=0
tol=0.0001
y1=0.5
y2=0.8
f1=runge_kutta("z","x*y**2",0,1,1,1,y1)
f2=runge_kutta("z","x*y**2",0,1,1,1,y2)
#print(f1,f2)
y3=secant(0.5,0.8,(f1-target),(f2-target))
#print(y3)
while 1:
    f3 = runge_kutta("z", "x * y**2", 0, 1, 100, 1, y3)
    if abs(f3 - target) < tol:
        print("Solution found: y(0) =", y3)
        break
    else:
        # Update y1, y2, f1, f2 for next iteration
        y1 = y2
        y2 = y3
        f1 = f2
        f2 = f3
        # Calculate new y3 using secant method
        y3 = secant(y1, y2, f1 - target, f2 - target)
import numpy as np
# Coefficient matrix
a = np.array([[4,-1,0,-1,0,0,0,0,0],
              [-1,4,-1,0,-1,0,0,0,0],
              [0,-1,4,0,0,-1,0,0,0],
              [-1,0,0,4,-1,0,-1,0,0],
              [0,-1,0,-1,4,-1,0,-1,0],
              [0,0,-1,0,-1,4,0,0,-1],
              [0,0,0,-1,0,0,4,-1,0],
              [0,0,0,0,-1,0,-1,4,-1],
              [0,0,0,0,0,-1,0,-1,4]])

# Right-hand side vector
b = np.array([1500,1000,1500,2000,0,2000,1500,1000,1500])

# Initial guess for the solution
x = np.array([939, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0])

# Number of iterations
N = 10
n=9
for k in range(N):
     print("Iteration number:", k + 1)
     for i in range(len(x)):
        sigma = 0
        for j in range(i):
            sigma += a[i][j] * x[j]
            sigma1=0
            for j in range(i+1,n):
                sigma1+=a[i][j]*x[j]
            x[i] = (1/ a[i][i]) * (b[i] - sigma-sigma1)
     print("solution=",x)
import numpy as np

# Coefficient matrix
a = np.array([[4, -1, 0, -1, 0, 0, 0, 0, 0],
              [-1, 4, -1, 0, -1, 0, 0, 0, 0],
              [0, -1, 4, 0, 0, -1, 0, 0, 0],
              [-1, 0, 0, 4, -1, 0, -1, 0, 0],
              [0, -1, 0, -1, 4, -1, 0, -1, 0],
              [0, 0, -1, 0, -1, 4, 0, 0, -1],
              [0, 0, 0, -1, 0, 0, 4, -1, 0],
              [0, 0, 0, 0, -1, 0, -1, 4, -1],
              [0, 0, 0, 0, 0, -1, 0, -1, 4]])

# Right-hand side vector
b = np.array([1500, 1000, 1500, 2000, 0, 2000, 1500, 1000, 1500])

# Initial guess for the solution
x = np.array([900, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Number of iterations
N = 20
n=9
# Perform Gauss-Seidel iteration
for k in range(N):
    print("Iteration number:", k + 1)
    for i in range(len(x)):
        sigma = 0
        for j in range(i):
            sigma += a[i][j] * x[j]
        sigma1 = 0
        for j in range(i + 1, n):
            sigma1 += a[i][j] * x[j]
        x[i] = (1 / a[i][i]) * (b[i] - sigma - sigma1)
    print("Solution:", x)
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
# BOUNDARY CONDITIONS....................................................................
u[:,0]=0
u[:,i]=0
#INITIAL CONDITIONS.............................................................................
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
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
def g(x):
    return 0.1*x*(4-x)
def f(x):
    return 0
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




import numpy as np
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
i=3
j=3
def f(x,y):
    return -(h**2)*81*x*y

u=np.zeros(shape=(i+1,j+1))
u_temp=np.zeros(shape=(i+1,j+1))
u[:,i]=100
u[0,:]=100
u[:,0]=0
u[j,:]=0
print(u)
u_temp=u.copy()
for it in range(50):
    u[:]=u_temp[:]
    for row in range(1, j):
        for col in range(1, i):
            u_temp[row, col] = 0.25 * (u[row+1, col] + u[row-1, col] + u[row, col+1] + u[row, col-1] +f((((j-row)*h)),(col*h) ))
    diff=np.abs(u-u_temp)
    if np.all(diff<10**(-6)):
        print(it)
        break

print(u)
import numpy as np
import math as m
a=float(input("enter the value of a="))
k=float(input("enter the value of k="))
h=float(input("enter the value of h="))
x_b=float(input("enter the boundary value of x="))
t_b=float(input("enter the boundary value of t="))
def f(x):
    return 1+2*x
def g(x):
    return 3-2*x
y=k/(a*h**2)
i=int(x_b/h)
j=int(t_b/k)
u=np.zeros(shape=(j+1,i+1))
u[:,0]=1
u[:,i]=1
for col in range(1,i):
    if (col*h<=0.5):
        u[j,col]=f(col*h)
    else:
        u[j,col]=g(col*h)
print(u)
    
for row in range(j-1,-1,-1):
    for col in range(1,i):
        u[row,col]=y*u[row+1,col+1] + (1-2*y)*u[row+1,col] + y*u[row+1,col-1]
print(u)
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
    return 2*x*(1-x)
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
print(u)'''
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














       
