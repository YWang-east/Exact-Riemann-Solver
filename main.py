from exact_Riemann import *
from approx_Riemann import *
import numpy as np

testcase = 4
L  = 1.0
RS = exact_Riemann_solver(L)

if (testcase==1):
    t  = 0.0002
    x0 = 0.5
    rho_l, ul, pl, gamma_l, pinf_l = 1.0, 800, 1e5, 1.4, 0
    rho_r, ur, pr, gamma_r, pinf_r = 2.0, 400, 4e5, 1.4, 0
    wl = np.array([rho_l, ul, pl, gamma_l, pinf_l])
    wr = np.array([rho_r, ur, pr, gamma_r, pinf_r])
    RS.init_shock_tube(t, x0, wl, wr)

elif(testcase==2): # water air shock-tube
    t  = 0.0002
    x0 = 0.7
    rho_l, ul, pl, gamma_l, pinf_l = 1e3, 0, 1e9, 6.12, 343e6
    rho_r, ur, pr, gamma_r, pinf_r = 1.0, 0, 1e5, 1.4, 0
    cv_l = 4181/gamma_l
    cv_r = 1005/gamma_r
    wl = np.array([rho_l, ul, pl, gamma_l, pinf_l])
    wr = np.array([rho_r, ur, pr, gamma_r, pinf_r])
    RS.init_shock_tube(t, x0, wl, wr, cv_l, cv_r)

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

elif(testcase==4):  # two interface - light bubble
    t  = 0.0004
    Ma = 1.2
    xs = 0.2
    xb = np.array([0.5, 1.1])
    rho_2, u2, p2, gamma_2, pinf_2 = 1,    0, 1e5 , 1.4, 0
    rho_1, u1, p1, gamma_1, pinf_1 = 1e3 , 0, 1e5 , 6.12, 343000000
    cv_1 = 4181/gamma_1
    cv_2 = 1005/gamma_2
    w1 = np.array([rho_1, u1, p1, gamma_1, pinf_1])     # ambient fluid
    w2 = np.array([rho_2, u2, p2, gamma_2, pinf_2])     # bubble fluid
    RS.init_shock_interface(t, Ma, xs, xb, w1, w2, cv_1, cv_2)

elif(testcase==5):  # two interface - heavy bubble
    t  = 0.0003
    Ma = 2.0    # shock Mach
    xs = 0.3    # initial shock location
    xb = np.array([0.4, 0.6])   # initial bubble position   
    rho_1, u1, p1, gamma_1, pinf_1 = 1.0,    0, 1e5 , 1.4 , 0
    rho_2, u2, p2, gamma_2, pinf_2 = 1000.0, 0, 1e5 , 6.12, 343000000
    w1 = np.array([rho_1, u1, p1, gamma_1, pinf_1])     # ambient fluid
    w2 = np.array([rho_2, u2, p2, gamma_2, pinf_2])     # bubble fluid
    RS.init_shock_interface(t, Ma, xs, xb, w1, w2)

# RS.plot_solutions()
RS.export_solutions()