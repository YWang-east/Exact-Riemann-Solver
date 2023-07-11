import numpy as np
import matplotlib.pyplot as plt

class exact_Riemann_solver(object):
	def __init__(self, L):
		self.L   = L

	def init_shock_tube(self, t, x0, wl: np.array, wr: np.array):
		self.t   = t
		self.x0  = x0
		self.wl  = wl 
		self.wr  = wr
		self.case = 'shock-tube'

	def init_shock_interface(self, t, Ma, xs, xb, w1, w2):
		"""
		w = [rho, u, p, gamma, pinf]
		"""
		w11      = np.zeros(5)
		w11[:] 	 = w1[:]	# post-shock states
		p_ratio  = 1 + 2*w1[3]/(w1[3]+1)*(Ma*Ma-1)
		r_ratio  = ((w1[3]-1)*p_ratio + w1[3]+1)/((w1[3]+1)*p_ratio + w1[3]-1)
		w11[0]   = w1[0]/r_ratio
		w11[1]   = 2/(w1[3]+1)*(Ma-1/Ma) * np.sqrt(w1[3]*(w1[2]+w1[4])/w1[0])
		w11[2]   = (w1[2]+w1[4])*p_ratio - w1[4]
		self.Vs  = Ma * np.sqrt(w1[3]*(w1[2]+w1[4])/w1[0])
		self.w1  = w1		# pre-shock ambient fluid
		self.w11 = w11		# post-shock ambient fluid
		self.w2  = w2		# pre-shock bubble fluid
		self.t   = t
		self.xs0 = xs		# initial position of shock
		self.xb0 = xb  		# initial position of bubble
		self.case = 'shock-interface'

	def plot_solutions(self):
		"""
		plots the exact solutions
		"""
		x, w = self.construct_sol(self.case)

		plt.rcParams['text.usetex'] = True
		plt.rcParams["font.family"] = "Times New Roman"
		fig = plt.figure(figsize=(12,8)) 
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)

        # top-left
		ax1.plot(x,w[0],c='k')
		ax1.set_ylabel(r'$\rho$', fontsize=15, fontweight = 'bold')
		ax1.grid()
    
        # top-right
		ax2.plot(x,w[1],c='k')
		ax2.set_ylabel(r'$u$', fontsize=15, fontweight = 'bold')
		ax2.grid()
    
        # bottom-left
		ax3.plot(x,w[2],c='k')
		ax3.set_ylabel(r'$p$', fontsize=15, fontweight = 'bold')
		ax3.grid()
		
		plt.show()

	def construct_sol(self, case):
		t, L = self.t, self.L

		N = 1000
		x = np.linspace(0, L, N)
		w = np.zeros((3, N))

		if (case=='shock-tube'):
			x0 = self.x0
			self.solve_RP(self.wl, self.wr)

			rl, ul, pl, gamma_l, pinf_l = self.wl[0], self.wl[1], self.wl[2], self.wl[3], self.wl[4]
			rr, ur, pr, gamma_r, pinf_r = self.wr[0], self.wr[1], self.wr[2], self.wr[3], self.wr[4]	
			al = np.sqrt(gamma_l*((pl+pinf_l)/rl))
			ar = np.sqrt(gamma_r*((pr+pinf_r)/rr))

			if(self.wave_l == 0):	# left shock
				x1 = x0 + t * self.V_shock_l
				x2 = x1
			else:				# left rarefaction
				x1 = x0 + t * self.V_head_l
				x2 = x0 + t * self.V_tail_l

			x3 = x0 + t * self.u_star	# contact discontinuity

			if(self.wave_r == 0):	# right shock
				x4 = x0 + t * self.V_shock_r
				x5 = x4
			else:				# right rarefaction
				x4 = x0 + t * self.V_tail_r
				x5 = x0 + t * self.V_head_r

			for i in range(N):
				if(x[i] < x1):
					w[0,i] = rl
					w[1,i] = ul
					w[2,i] = pl
				elif(x[i] > x1 and x[i] < x2):
					w[0,i] = rl * np.power(2/(gamma_l+1) + (gamma_l-1)/(gamma_l+1)/al*(ul - (x[i]-x0)/t), 2/(gamma_l-1))
					w[1,i] = 2/(gamma_l+1) * (al + (gamma_l-1)/2*ul + (x[i]-x0)/t)
					w[2,i] = (pl+pinf_l) * np.power(2/(gamma_l+1) + (gamma_l-1)/(gamma_l+1)/al*(ul - (x[i]-x0)/t), 2*gamma_l/(gamma_l-1)) - pinf_l
				elif(x[i] > x2 and x[i] < x3):
					w[0,i] = self.r_star_l
					w[1,i] = self.u_star
					w[2,i] = self.p_star
				elif(x[i] > x3 and x[i] < x4):
					w[0,i] = self.r_star_r
					w[1,i] = self.u_star
					w[2,i] = self.p_star	
				elif(x[i] > x4 and x[i] < x5):
					w[0,i] = rr * np.power(2/(gamma_r+1) - (gamma_r-1)/(gamma_r+1)/ar*(ur - (x[i]-x0)/t), 2/(gamma_r-1))
					w[1,i] = 2/(gamma_r+1) * (-ar + (gamma_l-1)/2*ur + (x[i]-x0)/t)
					w[2,i] = (pr+pinf_r) * np.power(2/(gamma_r+1) - (gamma_r-1)/(gamma_r+1)/ar*(ur - (x[i]-x0)/t), 2*gamma_r/(gamma_r-1)) - pinf_r
				elif(x[i] > x5):
					w[0,i] = rr
					w[1,i] = ur
					w[2,i] = pr

		elif (case=='shock-interface'):
			# pre-shock ambient fluid
			w[0, :] = self.w1[0]
			w[1, :] = self.w1[1]
			w[2, :] = self.w1[2]

			w[0, np.logical_and(x>self.xb0[0], x<self.xb0[1])] = self.w2[0]
			w[1, np.logical_and(x>self.xb0[0], x<self.xb0[1])] = self.w2[1]
			w[2, np.logical_and(x>self.xb0[0], x<self.xb0[1])] = self.w2[2]

			t1 = (self.xb0[0]-self.xs0)/self.Vs	# time of first collision

			if (t < t1):
				##	
				# before first collision
				##
				xs = self.xs0 + self.Vs * t		# position of initial shock
				# post-shock ambient fluid
				w[0, x<xs] = self.w11[0]
				w[1, x<xs] = self.w11[1]
				w[2, x<xs] = self.w11[2]
			else:
				##	
				# after first collision
				##			
				# new states after first collision
				self.solve_RP(self.w11, self.w2)
				w12 = np.array([self.r_star_l, self.u_star, self.p_star, self.w11[3], self.w11[4]])
				w21 = np.array([self.r_star_r, self.u_star, self.p_star,  self.w2[3],  self.w2[4]])

				xb1 = self.xb0[0] + self.u_star    * (t-t1)	# position of left interface
				xs  = self.xb0[0] + self.V_shock_r * (t-t1)	# position of transmitted shock
				if(self.wave_l==0):	# position of reflected shock 
					x10 = self.xb0[0] + self.V_shock_l * (t-t1)
				else:				# position of reflected rarefaction 	
					x10 = self.xb0[0] + self.V_head_l  * (t-t1)
					x11 = self.xb0[0] + self.V_tail_l  * (t-t1)

				t2 = t1 + (self.xb0[1]-self.xb0[0])/self.V_shock_r	# time of 2nd collision
				
				# solution region 1'
				w[0, x<x10] = self.w11[0]
				w[1, x<x10] = self.w11[1]
				w[2, x<x10] = self.w11[2]
				# solution region 1"
				if (self.wave_l==0):	# reflected wave is shock
					w[0, np.logical_and(x>x10, x<xb1)] = w12[0]
					w[1, np.logical_and(x>x10, x<xb1)] = w12[1]
					w[2, np.logical_and(x>x10, x<xb1)] = w12[2]
				else:					# reflected wave is rarefaction
					w[0, np.logical_and(x>x11, x<xb1)] = w12[0]
					w[1, np.logical_and(x>x11, x<xb1)] = w12[1]
					w[2, np.logical_and(x>x11, x<xb1)] = w12[2]

					rl, ul, pl, gamma_l, pinf_l = self.w11[0], self.w11[1], self.w11[2], self.w11[3], self.w11[4]
					al = np.sqrt(gamma_l*(pl+pinf_l)/rl)
					ir = np.logical_and(x>x10, x<x11)
					x0 = self.xb0[0]
					w[0, ir] = rl * np.power(2/(gamma_l+1) + (gamma_l-1)/(gamma_l+1)/al*(ul - (x[ir]-x0)/(t-t1)), 2/(gamma_l-1))
					w[1, ir] = 2/(gamma_l+1) * (al + (gamma_l-1)/2*ul + (x[ir]-x0)/(t-t1))
					w[2, ir] = (pl+pinf_l) * np.power(2/(gamma_l+1) + (gamma_l-1)/(gamma_l+1)/al*(ul - (x[ir]-x0)/(t-t1)), 2*gamma_l/(gamma_l-1)) - pinf_l
				
				if (t < t2):
					# solution region 2'
					w[0, np.logical_and(x>xb1, x<xs)] = w21[0]
					w[1, np.logical_and(x>xb1, x<xs)] = w21[1]
					w[2, np.logical_and(x>xb1, x<xs)] = w21[2]	
				else:
					##	
					# after second collision
					##
					# new states after second collision
					self.solve_RP(w21, self.w1)
					w22 = np.array([self.r_star_l, self.u_star, self.p_star, 	  w21[3], 	   w21[4]])
					w13 = np.array([self.r_star_r, self.u_star, self.p_star,  self.w1[3],  self.w1[4]])

					xb2 = self.xb0[1] + self.u_star    * (t-t2)	# position of right interface
					xs  = self.xb0[1] + self.V_shock_r * (t-t2)	# position of transmitted shock
					if(self.wave_l==0):	# position of reflected shock 
						x20 = self.xb0[1] + self.V_shock_l * (t-t2)
						# time of third collision
						t3  = (self.xb0[1] - self.xb0[0] + w12[1]*t1 - self.V_shock_l*t2)/(w12[1]-self.V_shock_l)
					else:				# position of reflected rarefaction 	
						x20 = self.xb0[1] + self.V_head_l  * (t-t2)
						x21 = self.xb0[1] + self.V_tail_l  * (t-t2)
						# time of third collision
						t3  = (self.xb0[1] - self.xb0[0] + w12[1]*t1 -  self.V_head_l*t2)/(w12[1]- self.V_head_l)
					
					# solution region 2'
					w[0, np.logical_and(x>xb1, x<x20)] = w21[0]
					w[1, np.logical_and(x>xb1, x<x20)] = w21[1]
					w[2, np.logical_and(x>xb1, x<x20)] = w21[2]
					# solution region 1"'
					w[0, np.logical_and(x>xb2, x<xs)]  = w13[0]
					w[1, np.logical_and(x>xb2, x<xs)]  = w13[1]
					w[2, np.logical_and(x>xb2, x<xs)]  = w13[2]
					# solution region 2"
					if (self.wave_l==0):	# reflected wave is shock
						w[0, np.logical_and(x>x20, x<xb2)] = w22[0]
						w[1, np.logical_and(x>x20, x<xb2)] = w22[1]
						w[2, np.logical_and(x>x20, x<xb2)] = w22[2]
					else:					# reflected wave is rarefaction
						w[0, np.logical_and(x>x21, x<xb2)] = w22[0]
						w[1, np.logical_and(x>x21, x<xb2)] = w22[1]
						w[2, np.logical_and(x>x21, x<xb2)] = w22[2]

						rl, ul, pl, gamma_l, pinf_l = self.w21[0], self.w21[1], self.w21[2], self.w21[3], self.w21[4]
						al = np.sqrt(gamma_l*(pl+pinf_l)/rl)
						ir = np.logical_and(x>x20, x<x21)
						x0 = self.xb0[1]
						w[0, ir] = rl * np.power(2/(gamma_l+1) + (gamma_l-1)/(gamma_l+1)/al*(ul - (x[ir]-x0)/(t-t1)), 2/(gamma_l-1))
						w[1, ir] = 2/(gamma_l+1) * (al + (gamma_l-1)/2*ul + (x[ir]-x0)/(t-t1))
						w[2, ir] = (pl+pinf_l) * np.power(2/(gamma_l+1) + (gamma_l-1)/(gamma_l+1)/al*(ul - (x[ir]-x0)/(t-t1)), 2*gamma_l/(gamma_l-1)) - pinf_l

					# TODO: solutions after 3rd collision not included
					if (t > t3): print('solutions not available!')

		return x, w
	
	def solve_RP(self, wl, wr):
		"""
		solve Riemann problem with left state (x<x0): wl=(rl,ul,pl); right state (x>x0): wr=(rr,ur,pr)
		"""
		rl, ul, pl, gamma_l, pinf_l = wl[0], wl[1], wl[2], wl[3], wl[4]
		rr, ur, pr, gamma_r, pinf_r = wr[0], wr[1], wr[2], wr[3], wr[4]	
		al = np.sqrt(gamma_l*((pl+pinf_l)/rl))
		ar = np.sqrt(gamma_r*((pr+pinf_r)/rr))

		# calculate p_star
		p_star = self.calc_p_star(rl,ul,pl,gamma_l,pinf_l,rr,ur,pr,gamma_r,pinf_r)

		# calculate u_star
		if (p_star < pl):
			if(p_star > pr):	# rarefaction-shock
				u_star = 0.5*((ur + ul) + \
						self.f(p_star, rr, pr, gamma_r, pinf_r) - \
					   	self.g(p_star, rl, pl, gamma_l, pinf_l)) 

			else:				# rarefaction-rarefaction
				u_star = 0.5*((ur + ul) - \
						self.g(p_star, rr, pr, gamma_r, pinf_r) - \
					   	self.g(p_star, rl, pl, gamma_l, pinf_l)) 
						
		else:
			if(p_star > pr):	# shock-shock
				u_star = 0.5*((ur + ul) + \
						self.f(p_star, rr, pr, gamma_r, pinf_r) - \
					   	self.f(p_star, rl, pl, gamma_l, pinf_l))
						
			else:				# shock-rarefaction
				u_star = 0.5*((ur + ul) - \
						self.f(p_star, rl, pl, gamma_l, pinf_l) - \
					   	self.g(p_star, rr, pr, gamma_r, pinf_r)) 
				
		# calculate r_star_l, r_star_r
		if (p_star < pl):				
			# left rarefaction
			r_star_l  = rl * np.power((p_star+pinf_l)/(pl+pinf_l), 1/gamma_l)
			a_star_l  = np.sqrt(gamma_l*((p_star+pinf_l)/r_star_l))
			V_head_l  = ul - al				# head velocity of rarefaction
			V_tail_l  = u_star - a_star_l	# tail velocity of rarefaction
			V_shock_l = 0
			wave_l    = 1

		else:
			# left shock
			r_star_l  = rl*(
							(2.0*	   gamma_l*pinf_l  + (gamma_l+1.0)*p_star + (gamma_l-1.0)*pl)/ \
							(2.0*(pl + gamma_l*pinf_l) + (gamma_l-1.0)*p_star + (gamma_l-1.0)*pl)
							)
			V_shock_l = ul - al * np.sqrt((gamma_l+1)/2/gamma_l * (p_star+pinf_l)/(pl+pinf_l) + (gamma_l-1)/2/gamma_l)
			V_head_l  = 0
			V_tail_l  = 0
			wave_l    = 0

		# determine wave types and speeds	
		if (p_star < pr):				
			# right rarefaction
			r_star_r  = rr * np.power((p_star+pinf_r)/(pr+pinf_r), 1/gamma_r)
			a_star_r  = np.sqrt(gamma_r*((p_star+pinf_r)/r_star_r))
			V_head_r  = ur + ar				# head velocity of rarefaction
			V_tail_r  = u_star + a_star_r	# tail velocity of rarefaction
			V_shock_r = 0
			wave_r    = 1

		else:
			# right shock
			r_star_r  = rr*(
							(2.0*      gamma_r*pinf_r  + (gamma_r+1.0)*p_star + (gamma_r-1.0)*pr)/ \
		   				   	(2.0*(pr + gamma_r*pinf_r) + (gamma_r-1.0)*p_star + (gamma_r-1.0)*pr)
							)
			V_shock_r = ur + ar * np.sqrt((gamma_r+1)/2/gamma_r * (p_star+pinf_r)/(pr+pinf_r) + (gamma_r-1)/2/gamma_r)
			V_head_r  = 0
			V_tail_r  = 0				
			wave_r    = 0

		self.r_star_l  = r_star_l
		self.r_star_r  = r_star_r
		self.u_star    = u_star
		self.p_star    = p_star
		self.wave_l    = wave_l
		self.wave_r    = wave_r
		self.V_shock_l = V_shock_l
		self.V_shock_r = V_shock_r
		self.V_head_l  = V_head_l
		self.V_tail_l  = V_tail_l
		self.V_head_r  = V_head_r
		self.V_tail_r  = V_tail_r	
	
	def calc_p_star(self,rl,ul,pl,gamma_l,pinf_l,rr,ur,pr,gamma_r,pinf_r):
		tol = 1e-6
		n   = 100

		# initial guess 
		p_star = 0.5*(pl+pr)

		# solve p_star using Newton-Raphson
		for i in range(n):
			p_star_new = p_star - self.eqn_p_star(p_star,rl,ul,pl,gamma_l,pinf_l,rr,ur,pr,gamma_r,pinf_r)/ \
								self.d_eqn_p_star(p_star,rl,ul,pl,gamma_l,pinf_l,rr,ur,pr,gamma_r,pinf_r)
			if (np.abs(p_star_new-p_star)/(p_star_new+p_star)/2 < tol): break
			if (i==n): print("Unable to find p star")
			p_star = p_star_new
	
		return p_star

	def eqn_p_star(self,p_star,rl,ul,pl,gamma_l,pinf_l,rr,ur,pr,gamma_r,pinf_r):
		"""
		equation to solve middle p
		"""
		# left rarefaction
		if (p_star < pl):
			if(p_star > pr):	# right shock
				return	self.f(p_star, rr, pr, gamma_r, pinf_r) + \
					   	self.g(p_star, rl, pl, gamma_l, pinf_l) + \
						ur - ul
			else:				# right rarefaction
				return	self.g(p_star, rr, pr, gamma_r, pinf_r) + \
					   	self.g(p_star, rl, pl, gamma_l, pinf_l) + \
						ur - ul
		# left shock
		else:
			if(p_star > pr):	# right shock
				return	self.f(p_star, rr, pr, gamma_r, pinf_r) + \
					   	self.f(p_star, rl, pl, gamma_l, pinf_l) + \
						ur - ul
			else:				# right rarefaction
				return	self.f(p_star, rl, pl, gamma_l, pinf_l) + \
					   	self.g(p_star, rr, pr, gamma_r, pinf_r) + \
						ur - ul
	
	def d_eqn_p_star(self,p_star,rl,ul,pl,gamma_l,pinf_l,rr,ur,pr,gamma_r,pinf_r):
		"""
		derivatives for Newton-Raphson method
		"""
		# left rarefaction
		if (p_star < pl):
			if(p_star > pr):	# right shock
				return	self.df(p_star, rr, pr, gamma_r, pinf_r) + \
					   	self.dg(p_star, rl, pl, gamma_l, pinf_l)
			else:				# right rarefaction
				return	self.dg(p_star, rr, pr, gamma_r, pinf_r) + \
					   	self.dg(p_star, rl, pl, gamma_l, pinf_l)
		# left shock
		else:
			if(p_star > pr):	# right shock
				return	self.df(p_star, rr, pr, gamma_r, pinf_r) + \
					   	self.df(p_star, rl, pl, gamma_l, pinf_l)
			else:				# right rarefaction
				return	self.df(p_star, rl, pl, gamma_l, pinf_l) + \
					   	self.dg(p_star, rr, pr, gamma_r, pinf_r)

	def f(self, p_star, r, p, gamma, pinf):
		A = 2/r/(gamma+1)
		B = (p+pinf)*(gamma-1.0)/(gamma+1.0)
		return (p_star - p) * np.sqrt(A/(p_star+pinf+B))
		
	def g(self, p_star, r, p, gamma, pinf):
		a = np.sqrt(gamma*((p+pinf)/r))
		return (2.0*a/(gamma-1.0)) * (np.power((p_star+pinf)/(p+pinf), (gamma-1.0)/(2.0*gamma)) - 1.0)
	
	def df(self, p_star, r, p, gamma, pinf):
		A = 2/r/(gamma+1)
		B = (p+pinf)*(gamma-1.0)/(gamma+1.0)
		return np.sqrt(A/(B+p_star+pinf))*(1.0 - ((p_star-p)/(2.0*(B+p_star+pinf))))
		
	def dg(self, p_star, r, p, gamma, pinf):
		a = np.sqrt(gamma*((p+pinf)/r))
		return (1.0/(r*a)) * np.power((p_star+pinf)/(p+pinf), -(gamma+1.0)/(2.0*gamma))
	
	def plot_hugoniot(self, wl, wr):
		"""
		plots the Hugoniot locus and integral curves
		"""
		N  = 400
		p  = np.linspace(0,4,N)
		ul_r1, ul_r2 = np.zeros(N), np.zeros(N)
		ul_s1, ul_s2 = np.zeros(N), np.zeros(N)
		ur_r1, ur_r2 = np.zeros(N), np.zeros(N)
		ur_s1, ur_s2 = np.zeros(N), np.zeros(N)

		for i in range(N):
			# integral curves for left state
			ul_r1[i] = wl[1] - self.g(p[i], wl[0], wl[2], wl[3], wl[4])
			ul_r2[i] = wl[1] + self.g(p[i], wl[0], wl[2], wl[3], wl[4])
			# hugoniot locus for left state
			ul_s1[i] = wl[1] - self.f(p[i], wl[0], wl[2], wl[3], wl[4])
			ul_s2[i] = wl[1] + self.f(p[i], wl[0], wl[2], wl[3], wl[4])

			# integral curves for right state
			ur_r1[i] = wr[1] - self.g(p[i], wr[0], wr[2], wr[3], wr[4])
			ur_r2[i] = wr[1] + self.g(p[i], wr[0], wr[2], wr[3], wr[4])
			# hugoniot locus for right state
			ur_s1[i] = wr[1] - self.f(p[i], wr[0], wr[2], wr[3], wr[4])
			ur_s2[i] = wr[1] + self.f(p[i], wr[0], wr[2], wr[3], wr[4])

		fig, ax = plt.subplots()
		ax.plot(p, ul_r1, 'b--')
		ax.plot(p, ul_r2, 'b--')
		ax.plot(p, ul_s1, 'b-')
		ax.plot(p, ul_s2, 'b-')
		ax.plot(p, ur_r1, 'r--')
		ax.plot(p, ur_r2, 'r--')
		ax.plot(p, ur_s1, 'r-')
		ax.plot(p, ur_s2, 'r-')
		plt.show()
	