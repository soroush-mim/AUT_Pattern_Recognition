import numpy as np
import matplotlib.pyplot as plt 



x = np.linspace(-30, 30, 15000)
plt.plot(x,-0.4*x-0.5 )

Y, X = np.mgrid[-30:30:300j, -30:30:300j]
plt.contour(X, Y, (63*X**2 - 128*X +36*Y**2 + 28*X*Y -172*Y + 236)/55 - 
            (25*X**2 + 20 * X + 25 *Y**2 -10*X*Y+140*Y+220)/36 +0.24  , levels=[0])
plt.grid()

plt.savefig('2b.png')