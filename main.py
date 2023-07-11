from exact_Riemann import *
from approx_Riemann import *
import numpy as np

testcase = 4
L  = 1.0
RS = exact_Riemann_solver(L)

if (testcase==1):
    t  = 0.25
    x0 = 0.5
    rho_l, ul, pl, gamma_l, pinf_l = 1,     0, 1.0, 1.4, 0
    rho_r, ur, pr, gamma_r, pinf_r = 0.125, 0, 0.1, 1.4, 0
    wl = np.array([rho_l, ul, pl, gamma_l, pinf_l])
    wr = np.array([rho_r, ur, pr, gamma_r, pinf_r])
    RS.init_shock_tube(t, x0, wl, wr)

elif(testcase==2):
    t  = 0.1
    x0 = 0.5
    rho_l, ul, pl, gamma_l, pinf_l = 1.241, 0, 2.753    , 1.4, 0
    rho_r, ur, pr, gamma_r, pinf_r = 0.991, 0, 3.059e-4 , 5.5, 1.505
    wl = np.array([rho_l, ul, pl, gamma_l, pinf_l])
    wr = np.array([rho_r, ur, pr, gamma_r, pinf_r])
    RS.init_shock_tube(t, x0, wl, wr)

elif(testcase==3):  # one interface
    t  = 0.2
    Ma = 1.95254
    xs = 0.4
    xb = np.array([0.5, 1.1*L])
    rho_2, u2, p2, gamma_2, pinf_2 = 1.0, 0, 1.0 , 1.4, 0
    rho_1, u1, p1, gamma_1, pinf_1 = 5.0, 0, 1.0 , 4.0, 1.0
    w1 = np.array([rho_1, u1, p1, gamma_1, pinf_1])     # ambient fluid
    w2 = np.array([rho_2, u2, p2, gamma_2, pinf_2])     # bubble fluid
    RS.init_shock_interface(t, Ma, xs, xb, w1, w2)

elif(testcase==4):  # two interface
    t  = 0.22
    Ma = 1.95254
    xs = 0.3
    xb = np.array([0.4, 0.7])
    rho_2, u2, p2, gamma_2, pinf_2 = 1.0, 0, 1.0 , 1.4, 0
    rho_1, u1, p1, gamma_1, pinf_1 = 5.0, 0, 1.0 , 4.0, 1.0
    w1 = np.array([rho_1, u1, p1, gamma_1, pinf_1])     # ambient fluid
    w2 = np.array([rho_2, u2, p2, gamma_2, pinf_2])     # bubble fluid
    RS.init_shock_interface(t, Ma, xs, xb, w1, w2)

RS.plot_solutions()

# RS.plot_hugoniot()

# approx_RS = approx_Riemann_solver(gamma_l, pinf_l, gamma_r, pinf_r, wl, wr)
# approx_RS.plot_solutions(L,x0,0.1)