# python program for bisection method :)-
import math as m
def f(x):
    return 10**x+x-4
a=int(input("enter first initial guess="))
b=int (input("enter second initial guess="))
if f(a)*f(b)>0:
    print("bisection method fails")
else:
    n=0
    while n<=5:
        c=(a+b)/2
        n=n+1
        if f(a)*f(c)<0:
            b=c
        else:
            a=c
        print("root of the given equation=",c)



# python program for ruag falsi method :)-
def f(x):
    return x**4-x-10
a=int(input("enter the first intial guess="))
b=int(input("enter the second initial guess="))
if f(a)*f(b)>0:
    print("ruag falsi method fails")
else:
    n=0
    while n<5:
        c=(a*f(b)-b*f(a))/(f(b)-f(a))
        n=n+1
        if f(a)*f(c)<0:
            b=c
        else:
            a=c
        print("root of the equation=",c)


        
# python program for secant method :)-
def f(x):
    return x**4-x-10
i=1
x0=int(input("enter the first initial guess="))
x1=int(input("enter the second initial guess="))
n=0
while n<5:
    n=n+1
    xi=(x0*f(x1)-x1*f(x0))/(f(x1)-f(x0))
    x0=x1
    x1=xi
    print("root of the equation=",xi)



        
# python program for newton raphson method :)-
def f(x):
    return x**4-4*x**3+7*x**2-5*x-2
def df(x):
    return 4*x**3-12*x**2+14*x-5
x0=float(input("enter your initial guess="))
n=0
while n<5:
    n=n+1
    xi=x0 - (f(x0)/df(x0))
    x0=xi
    print("root of the equation=",xi)

#another code for newton raphson method :)-   
def newtonmethod(func,funcderiv,x0):
    def f(x):
        f=eval(func)
        return f
    def df(x):
        df=eval(funcderiv)
        return df
    n=0
    while n<5:
            n=n +1
            xi=x0- (f(x0)/df(x0))
            x0=xi
            print("root of the equation =",xi)
newtonmethod("x**2-8","2*x",3)



# pyhon program for golden section search for optimization:)-
def f(x):
    return 2*m.sin(x)- x**2/10
xl=int(input("enter the first interval="))
xu=int(input("enter the second interval="))
n=0
while n<5:
    n=n+1
    d=0.618*(xu-xl)
    x1=xl+d
    x2=xu-d
    if f(x1)>f(x2):
        xl=x2
        print("solution=",x1)
    else:
        xu=x1
        print("solution=",x2)


        

#bisection method for optimization in 1D :)-
def f(x):
    return 12*x-2*x**2
def df(x):
    return 12-4*x
a=float(input("enter first initial guess="))
b=float(input("enter the second initial guess="))
n=0
while n<5:
    n=n+1
    c=(a+b)/2
    if df(c)>0:
        a=c
    elif df(c)<0:
        b=c
    else:
        print ("this is the  exact maxima")
        exit()
    print("maxima=",c)


    
#python program for successive over relaxation method:)-
import numpy as np
#coefficient matrix
a=np.array([[45,2,3],[-3,22,2],[5,1,20]])
# right hand side vector
b=np.array([58,47,67])
# initial guess for the solution
x=np.array([0.0,0.0,0.0])
# number of iteration
N=10
n=3
w=float(input("enter the value of w="))
for k in range(N):
    print("iteration number=",k+1)
    for i in range(n):
        sigma=0
        for j in range(i):
            sigma+=a[i][j]*x[j]
        sigma1=0
        for j in range(i+1,n):
            sigma1+=a[i][j]*x[j]
        x[i]=(1-w)*x[i] + (w/a[i][i])*(b[i]-sigma-sigma1)
    print("solution=",x)


# python complete program for gauss siedel method :)-
import numpy as np
indicator=1
temp=1
# Coefficient matrix
a = np.array([[45,2,3],
              [-3,22,2],
              [5,1,20]])

# Right-hand side vector
b = np.array([58,47,67])

# Initial guess for the solution
x = np.array([0.0, 0.0, 0.0])

# Number of iterations
N = 10
n=3
# necessary condition checking
for i in range(n):
    if a[i][i]==0:
        print("iterative method fails")
        indicator =0
# convergence checking
for i in range(n):
    sum1=0

    for j in range(n):
        if j!=i:
            sum1=sum1+abs(a[i][j])
        
                 
    if abs(a[i][i])<sum1:
        temp=0
        break
    
if temp==1:
    print("the solution converges")
else:
    print("the solution does not converges")
# Perform iterations
if temp==1 and indicator==1 :
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



#python code for jacobi method :)-
import numpy as np
# Coefficient matrix
a = np.array([[26.0,2.0,2.0],
              [3.0,27.0,1.0],
              [2.0,3.0,17.0]])

# Right-hand side vector
b = np.array([12.6,-14.3,6.0])

# Initial guess for the solution
x = np.array([0.0, 0.0, 0.0])

# Number of iterations
N = 10

# Perform iterations
for iteration in range(N):
    print("Iteration number:", iteration + 1)
    for i in range(len(x)):
        sigma = 0
        for j in range(len(x)):
            if j != i:
                sigma += a[i][j] * x[j]
        x[i] = (1/ a[i][i]) * (b[i] - sigma)
    print("Solution:", x)


#python program for cholesky decomposition:)-
import math as m
import numpy as np
a=np.array([[25,15,-5],[15,18,0],[-5,0,11]])
b=np.array([35,33,6])
l=np.array([[0,0,0],[0,0,0],[0,0,0]])
l1=np.array([[0,0,0],[0,0,0],[0,0,0]])
x=np.array([0,0,0])
d=np.array([0,0,0])
for i in range(len(x)):
    for j in range(len(x)):
                   if i<j:
                       l[i][j]=0
                   elif i==j:
                       l[i][j]=m.sqrt(a[j][j]-sum(l[i][k]**2 for k in range(j)))
                   else:
                       l[i][j]=(a[i][j]-sum(l[i][k]*l[j][k] for k in range(j)))/l[j][j]
                  
print(l)
for i in range(len(x)):
    for j in range(len(x)):
        l1[i][j]=l[j][i]
print(l1)
d=np.linalg.solve(l,b)
x=np.linalg.solve(l1,d)
print(x)



# python program for lagrange interpolation :)-
import numpy as np
import matplotlib.pyplot as pl
x=[1,3,5,7,8,9,10,12,13]
y=[50,-30,-20,20,5,1,30,80,-10]
pl.plot(x,y,marker ='o',color='r',ls=' ',markersize=10)
n=len(x)
prange=np.linspace(min(x),max(x),100000)
def f(o):
    sum1 =0
    for i in range(n):
        prod=y[i]
        for j in range(n):
            if i!=j:
                prod=prod*(o-x[j])/(x[i]-x[j])
        sum1 =sum1+prod
    return sum1
pl.plot(prange,f(prange),color='g')
pl.show()



# python program for newton divided difference formula for interpolation :)-
import numpy as np
import matplotlib.pyplot as pl
x=[-5,-4,-3,-2,-1,0,1,2,3,4,5]
y=[0,0,0.1,0.3,0.7,1,0.7,0.3,0.1,0,0]
pl.plot(x,y,color='r',marker='o',ls=' ',markersize=10)
prange=np.linspace(min(x),max(x),1000)
n=len(x)
def diff(a,b):
    if a==0:
        return y[b]
    else:
        return (diff(a-1,b)-diff(a-1,a-1))/(x[b]-x[a-1])
def f(o):
    yres=0
    for p in range(len(x)):
        prodres=diff(p,p)
        for q in range(p):
            prodres=prodres*(o-x[q])
        yres=yres+prodres
    return yres
pl.plot(prange,f(prange))
pl.show()


#python program for spline interpolation :)-
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Generate some sample data
x = np.array([0, 1, 2, 3, 4, 5,6])
y = np.array([0, 1, 4, 9, 16, 25,36])

# Perform quadratic spline interpolation
f=interp1d(x,y,kind='quadratic')

# Define the points at which you want to interpolate
x_interp = np.linspace(min(x),max(x),50)
y_interp = f(x_interp)

# Plot the original data and the interpolated curve
plt.plot(x, y, 'o',ls=' ',color='r',markersize=10 )
plt.plot(x_interp, y_interp,color='g')
plt.show()



#python program for mid point method  :)-
import numpy as np
a=int(input("enter the lower bound ="))
b=int(input("enter the upper bound ="))

def f(x):
    return x**2
n=int(input("enter the number of interval="))
h=(b-a)/n
if h>0:
    sum1=0
    for i in range(n):
        sum1=sum1+h*(f(a+(i+0.5)*h))
print (sum1)



#python code for tripozoidal rule :)-
import numpy as np
a=int(input("enter the lower bound="))
b=int(input("enter the upper bound="))
n=int(input("enter the numbe of interval="))
h=(b-a)/n
def f(x):
    return x**2
if n>0:
    sum1=f(a)+f(b)
    sum2=0
    for i in range(1,n):
        x_i= a+i*h
        sum2=sum2 + 2*f(x_i)
print((h*(sum1+sum2))/2)



#python program for simpson 1/3rd rule :)-
a=int(input("enter the lower bound ="))
b=int(input("enter the upper bound ="))
n=int(input("enter the number of interval ="))
h=(b-a)/n
sum1=0
def f(x):
    return x**3
for i in range(1,n,2):
    x_i=a+i*h
    sum1=sum1 +4*f(x_i)
sum2=0
for j in range(2,n,2):
    x_j=a+j*h
    sum2=sum2+2*f(x_j)
sum3=f(a)+f(b)
print((h*(sum1+sum2+sum3)/3))


#python program for simpson 3/8 rule :)-
a=float(input("enter the lower value of interval="))
b=float(input("enter the upper value of the interval="))
n=int(input("enter the number of interval="))
h=(b-a)/n
def f(x):
    return x**3
sum1=f(a)+f(b)
sum2=0
sum3=0
for i in range(1,n):
    x_a=a+i*h
    if i%3==0 :
        sum2=sum2+f(x_a)
    else:
        sum3=sum3+f(x_a)
print((3*h/8)*(sum1+2*sum2+3*sum3))
    
    




#python program for implementing euler method to solve 1st order ODE :)-
a=int(input("enter an integer="))
b=int (input("enter the upper interval="))
n=int(input("enter the number of interval="))
h=(b-a)/n
def df(x,y):
    return (1/x**2) - (y/x) - (y)**2
y0=-1
for i in range(n):
    x_a=a+i*h
    y1=y0+h*(df(x_a,y0))
    y0=y1
    print("solution=",y1)

    
# example 2 :)-
a=int(input("enter an integer="))
b=int (input("enter the upper interval="))
n=int(input("enter the number of interval="))
h=(b-a)/n
def df(x,y):
    return (-x*y)+(1/y**2)
y0=1
for i in range(n):
    x_a=a+i*h
    y1=y0+h*(df(x_a,y0))
    y0=y1
    print("solution=",y1)


    
#python program to implement the runge kutta method to solve ODE :)-
a=float(input("enter the lower bound of interval="))
b=float(input("enter the upper bound of interval="))
n=int(input("enter the number of intervals="))
y0=float(input("enter the initial value of function="))
h=(b-a)/n
def df(x,y):
    return x+y
for i in range(n):
    x_a=a+i*h
    k1=h*df(x_a,y0)
    print(k1)
    k2=h*df(x_a+ (h/2),y0 + (k1)/2)
    print(k2)
    k3=h*df(x_a + (h/2),y0 + (k2)/2)
    print(k3)
    k4=h*df(x_a + h,y0 + k3)
    print(k4)
    y1=y0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    print("value of function=",y1)
    y0=y1

    
#example :)-
a=float(input("enter the lower bound of interval="))
b=float(input("enter the upper bound of interval="))
n=int(input("enter the number of intervals="))
y0=float(input("enter the initial value of function="))
h=(b-a)/n
def df(x,y):
    return (y**2 - x**2)/(y**2 + x**2)
for i in range(n):
    x_a=a+i*h
    k1=h*df(x_a,y0)
    print(k1)
    k2=h*df(x_a+ (h/2),y0 + (k1)/2)
    print(k2)
    k3=h*df(x_a + (h/2),y0 + (k2)/2)
    print(k3)
    k4=h*df(x_a + h,y0 + k3)
    print(k4)
    y1=y0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    print("value of function=",y1)
    y0=y1


#python program for milne predictor and simpson corrector method to solve ODE :)-
import numpy as np
import math as m
a=float(input("enter the initial value of x="))
b=float(input("enter the upper bound of x="))
n=int(input("enter the number of interval="))
y0=int(input("enter the value of y="))
h=(b-a)/n
x_a=np.zeros(n+1)
ys=np.zeros(n+1)
ys[0]=y0
def df(x,y):
    return x*y + y**2
for i in range(n):
    x_a[i]=a+i*h
    k1=h*(df(x_a[i],ys[i]))
    k2=h*(df(x_a[i] + (h/2),ys[i]+(k1/2)))
    k3=h*(df(x_a[i] + (h/2),ys[i]+(k2/2)))
    k4=h*(df(x_a[i] +h,ys[i]+k3))
    ys[i+1]= ys[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    print(ys[i+1])
#predictor formula
for i in range(1):
    x_a[4]=a+n*h
    ys[i+4]=ys[i] + (4*h/3)*(2*df(x_a[i+1],ys[i+1]) -df(x_a[i+2],ys[i+2]) +2*df(x_a[i+3],ys[i+3]))
    print("predicted solution y4=",ys[i+4])
    while 1 :
         y2=ys[i+4]
         ys[i+4]= ys[i+2] + (h/3)*(df(x_a[i+2],ys[i+2]) + 4*df(x_a[i+3],ys[i+3]) + df(x_a[i+4],ys[i+4]))
         if ys[i+4]==y2:
               print("corrector solution y4=",ys[i+4])
               break


#python program for adam bashforth predictor corrector method :)-         
import numpy as np
import math as m
a=float(input("enter the initial value of x="))
b=float(input("enter the upper bound of x="))
n=int(input("enter the number of interval="))
y0=int(input("enter the value of y="))
h=(b-a)/n
x_a=np.zeros(n+1)
ys=np.zeros(n+1)
ys[0]=y0
def df(x,y):
    return x - y**2
for i in range(n):
    x_a[i]=a+i*h
    k1=h*(df(x_a[i],ys[i]))
    k2=h*(df(x_a[i] + (h/2),ys[i]+(k1/2)))
    k3=h*(df(x_a[i] + (h/2),ys[i]+(k2/2)))
    k4=h*(df(x_a[i] +h,ys[i]+k3))
    ys[i+1]= ys[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    print(ys[i+1])
#predictor formula
for i in range(1):
    x_a[4]=a+n*h
    ys[i+4]=ys[i+3] + (h/24)*(55*df(x_a[i+3],ys[i+3]) -59*df(x_a[i+2],ys[i+2]) +37*df(x_a[i+1],ys[i+1]) - 9*df(x_a[i],ys[i]))
    print("predicted solution y4=",ys[i+4])
    while 1 :
         y2=ys[i+4]
         ys[i+4]= ys[i+3] + (h/24)*(df(x_a[i+4],ys[i+4]) + 19*df(x_a[i+3],ys[i+3]) -5* df(x_a[i+2],ys[i+2]) +9*df(x_a[i+1],ys[i+1]))
         if ys[i+4]==y2:
               print("corrected solution y4=",ys[i+4])
               break



#python program to solve system of ODE using runge kutta method of order 4 :)-
a=float(input("enter the lower value ="))
b=float(input("enter the upper value="))
n=int(input("enter the number of intervals="))
y0=float(input("enter the initial value of y="))
z0=float(input("enter the initial value of z="))
h=(b-a)/n
def df(x,y,z):
    return 1+x*z
def df1(x,y,z):
    return -x*y
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
    print("value of y1=",y1)
    print("value of z1=",z1)
    y0=y1
    z0=z1


    
#python program to solve second order differential equation :)-
a=float(input("enter the lower value ="))
b=float(input("enter the upper value="))
n=int(input("enter the number of intervals="))
y0=float(input("enter the initial value of y="))
z0=float(input("enter the initial value of z="))
h=(b-a)/n
def df1(x,y,z):
    return z
def df(x,y,z):
    return x*z**2 - y**2
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
    print("value of y1=",y1)
    print("value of z1=",z1)
    y0=y1
    z0=z1


#python program to solve second order differential equation :)-
def runge_kutta(func,derivfunc,a,b,n,y0,z0):
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
         print("value of y1=",y1)
         print("value of z1=",z1)
         y0=y1
         z0=z1
    return y1
        
runge_kutta("z","x*z**2-y**2",0,0.2,1,1,0)

         

#python program to implement shooting method :)-
#value at x=2
target=8
#initial guesse of z at x=0
y1=1
y2=4
f1=runge_kutta("z","2*x+x*y-x*z",0,2,1,1,y1)
f2=runge_kutta("z","2*x+x*y-x*z",0,2,1,1,y2)
print(f1,f2)
#interpolation
y3=y2+(y1-y2)*(target-f2)/(f1-f2)
print(runge_kutta("z","2*x+x*y-x*z",0,2,1,1,y3))

#value at x=
target=0
#initial guesse of z at x=0
y1=0
y2=2
f1=runge_kutta("z","-1-y",0,1,1,0,y1)
f2=runge_kutta("z","-1-y",0,1,1,0,y2)
#interpolation
y3=y2+(y1-y2)*(target-f2)/(f1-f2)
runge_kutta("z","-1-y",0,1,4,0,y3)
import numpy as np
def sor(a,b,x,n,N,w):
    for k in range(N):
        print("iteration number=",k+1)
        for i in range(n):
            sigma=0
            for j in range(i):
                sigma+=a[i][j]*x[j]
            sigma1=0
            for j in range(i+1,n):
                sigma1+=a[i][j]*x[j]
            x[i]=(1-w)*x[i] + (w/a[i][i])*(b[i]-sigma-sigma1)
        print("solution=",x)



#python program to solve boundary value ODE by fixed difference method :)-
import numpy as np
def solve_bvp_fd(f,p,r,a,b,alpha,beta,n):
    h=(b-a)/n
    x=np.linspace(a,b,n+1)
    A=np.zeros((n+1,n+1))
    rhs=np.zeros(n+1)
    for i in range(1,n):
        A[i, i-1] = 1 /(h**2) -  f(x[i])/(2*h)
        A[i, i] = -2 /(h**2) +  p(x[i])
        A[i, i+1] = 1 /(h**2) + f(x[i])/2*h
        rhs[i] = r(x[i])
    A[0,0]=1
    A[n,n]=1
    rhs[0]=alpha
    rhs[n]=beta
    print(A)
    print(rhs)
    sol=np.linalg.solve(A,rhs)
    return sol
def f(x):
    return 0
def p(x):
    return -1
def r(x):
    return x
a = 0.0  # left boundary
b = 1.0  # right boundary
alpha = 0.0  # y(a)
beta =0.0  # y(b)
n = 4  # number of subintervals
# Solve the BVP using finite differences
print(solve_bvp_fd(f,p,r, a, b, alpha, beta, n))




#python program to solve mixed boundary condition ODE by finite difference method :)-
import numpy as np
def solve_bvp_fd(f,p,r,a,b,alpha,beta,n):
    h=(b-a)/n
    x=np.linspace(a,b,n+1)
    A=np.zeros((n+1,n+1))
    rhs=np.zeros(n+1)
    for i in range(1,n):
        A[i, i-1] = 1 /(h**2) -  f(x[i])/(2*h)
        A[i, i] = -2 /(h**2) +  p(x[i])
        A[i, i+1] = 1 /(h**2) + f(x[i])/2*h
        rhs[i] = r(x[i])
    A[0,0]=1
    A[n,n-1]=(1/h**2 + f(x[i])/2*h)*(beta*2*h) + (2/h**2)
    A[n,n]=-2 /(h**2) +  p(x[i])
    print(A)
    rhs[0]=alpha
    rhs[n]=r(x[n])
    print(rhs)
    sol=np.linalg.solve(A,rhs)
    return sol
#f(x) is a function multiplied with df(x)
def f(x):
    return 5
#p(x) is a function multiplied with y
def p(x):
    return 4
#r(x) is a function on the rhs
def r(x):
    return 1
a = 0.0  # left boundary
b = 1.0  # right boundary
alpha = 1.0  # y(a)
beta =0.0  # y'(b)
n = 10  # number of subintervals
# Solve the BVP using finite differences
print(solve_bvp_fd(f,p,r, a, b, alpha, beta, n))








                          
         

    
   



   

  
