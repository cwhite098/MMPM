from Scientific_Computing import ode_solvers as ode
from Scientific_Computing import plots as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import fsolve

mpl.rcParams['text.usetex'] = False  # not really needed
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})

def root_finding_problem(x, *args):
    params = args[0]
    [dZ, dY] = calcium_model_reduced(x,[0],params)
    return [dZ, dY]


def get_stability(point, params):

    # get the location of the equilibrium point
    Z = point[0]
    Y = point[1]

    # Unpack the parameters
    V_M2 = params['V_M2']
    V_M3 = params['V_M3']
    K_2 = params['K_2']
    K_A = params['K_A']
    K_R = params['K_R']
    n = params['n']
    m = params['m']
    p = params['p']
    k_f = params['k_f']
    k = params['k']

    # Calculate the derivatives
    dfdz = -V_M2*( ((K_2**n + Z**n)*n*Z**(n-1) - n*Z**(2*n-1))/((K_2**n + Z**n)**2) ) + V_M3*((Y**m)/(K_R**m + Y**m))*( ((K_A**p + Z**p)*p*Z**(p-1) - p*Z**(2*p-1))/((K_A**p + Z**p)**2) ) - k
    dfdy = V_M3*((Z**p)/(K_A**p + Z**p))*( ((K_R**m + Y**m)*m*Y**(m-1) - m*Y**(2*m-1))/((K_R**m + Y**m)**2) ) + k_f

    dgdz = V_M2*( ((K_2**n + Z**n)*n*Z**(n-1) - n*Z**(2*n-1))/((K_2**n + Z**n)**2) ) - V_M3*((Y**m)/(K_R**m + Y**m))*( ((K_A**p + Z**p)*p*Z**(p-1) - p*Z**(2*p-1))/((K_A**p + Z**p)**2) )
    dgdy = -V_M3*((Z**p)/(K_A**p + Z**p))*( ((K_R**m + Y**m)*m*Y**(m-1) - m*Y**(2*m-1))/((K_R**m + Y**m)**2) ) + k_f

    # Form the Jacobian
    J = np.array([[dfdz,dfdy],[dgdz,dgdy]])

    # Calculate the eignvalues and vectors
    w,v = np.linalg.eig(J)
    print('Eigenvalues: ',w)

    return v, w

def calcium_model_reduced(x, t, params):
    # Unpack Z and Y
    Z = x[0]
    Y = x[1]
    
    # Params for v2 and v3
    V_M2 = params['V_M2']
    V_M3 = params['V_M3']
    K_2 = params['K_2']
    K_A = params['K_A']
    K_R = params['K_R']
    n = params['n']
    m = params['m']
    p = params['p']
    # Calculating v_2 and v_3
    v_2 = V_M2 * ((Z**n)/(K_2**n + Z**n))
    v_3 = V_M3 * ((Y**m)/(K_R**m + Y**m)) * ((Z**p)/(K_A**p + Z**p))

    # Params for freq encoding
    v_p = params['v_p']
    W_T = params['W_T']
    K_1 = params['K_1']
    K_2_2 = K_1
    K_a = params['K_a']
    V_MK = params['V_MK']
    # Calculating v_k
    v_K = V_MK * (Z/(K_a + Z))

    # Other params
    v_0 = params['v_0']
    v_1 = params['v_1']
    beta = params['beta']
    k_f = params['k_f']
    k = params['k']

    # Calculating derivatives
    dZ = v_0 + (v_1*beta) - v_2 + v_3 + (k_f*Y) - (k*Z)
    dY = v_2 - v_3 - (k_f*Y)

    return [dZ, dY]


def calcium_model(x, t, params):

    # Unpack Z and Y
    Z = x[0]
    Y = x[1]
    W_star = x[2]
    
    # Params for v2 and v3
    V_M2 = params['V_M2']
    V_M3 = params['V_M3']
    K_2 = params['K_2']
    K_A = params['K_A']
    K_R = params['K_R']
    n = params['n']
    m = params['m']
    p = params['p']
    # Calculating v_2 and v_3
    v_2 = V_M2 * ((Z**n)/(K_2**n + Z**n))
    v_3 = V_M3 * ((Y**m)/(K_R**m + Y**m)) * ((Z**p)/(K_A**p + Z**p))

    # Params for freq encoding
    v_p = params['v_p']
    W_T = params['W_T']
    K_1 = params['K_1']
    K_2_2 = K_1
    K_a = params['K_a']
    V_MK = params['V_MK']
    # Calculating v_k
    v_K = V_MK * (Z/(K_a + Z))

    # Other params
    v_0 = params['v_0']
    v_1 = params['v_1']
    beta = params['beta']
    k_f = params['k_f']
    k = params['k']

    # Calculating derivatives
    dZ = v_0 + (v_1*beta) - v_2 + v_3 + (k_f*Y) - (k*Z)
    dY = v_2 - v_3 - (k_f*Y)
    dWstar = (v_p/W_T)*((v_K/v_p)*((1-W_star)/(K_1 + 1 - W_star)) - (W_star)/(K_2_2 + W_star))

    return [dZ, dY, dWstar]



def obtain_sol(X0, t, n, m, p, beta, k, v_0):

    # Get numerical Solution
    sol = ode.solve_ode('rk4', calcium_model, t, X0,v_0=v_0,
                                                    v_1=7.3,
                                                    beta=beta,
                                                    k_f=1,
                                                    k=k,
                                                    V_M2=65,
                                                    V_M3=500,
                                                    K_2=1,
                                                    K_A=0.9,
                                                    K_R=2,
                                                    n=n,
                                                    m=m,
                                                    p=p,
                                                    v_p=5, # W* params
                                                    W_T=1,
                                                    K_1=0.1,
                                                    K_a=2.5,
                                                    V_MK=40)

    print(sol)

    #p.plot_solution(t, sol,'Time /s', 'Z and Y and W*', 'Calcium Model Solution')                                                

    return sol


def phase_plot(Z, Y, ax, n, m, p, beta, k, v_0):

    # Plot the trajectory provided
    ax.plot(Z, Y, '--', label='Trajectory', zorder=9 )
    ax.set_xlabel(r'$Z$')
    ax.set_ylabel(r'$Y$')
    ax.set_title('Phase Plane')

    # Use the same params used in the trajectory
    params= {'v_0':v_0,
            'v_1':7.3,
            'beta':beta,
            'k_f':1,
            'k':k,
            'V_M2':65,
            'V_M3':500,
            'K_2':1,
            'K_A':0.9,
            'K_R':2,
            'n':n,
            'm':m,
            'p':p,
            'v_p':5, # W* params
            'W_T':1,
            'K_1':0.1,
            'K_a':2.5,
            'V_MK':40}

    # Plot the vector field
    Z, Y = np.meshgrid(np.linspace(Z.min()-0.2, Z.max()+0.2, 30), np.linspace(Y.min()-0.2, Y.max()+0.2, 30))
    [dZ, dY] = calcium_model_reduced([Z, Y], [0], params)
    dZ = dZ / np.sqrt(dZ**2 + dY**2)
    dY = dY / np.sqrt(dZ**2 + dY**2)
    ax.quiver(Z, Y, dZ, dY, alpha=0.5, units='inches', width=0.009 )#, headlength=15*5, headwidth=9*5, headaxislength=13.5*5, minshaft=1.5*5)

    # Plot the nullclines using contours
    Z, Y = np.meshgrid(np.linspace(Z.min(), Z.max(), 1000), np.linspace(Y.min(), Y.max(), 1000))
    [dZ, dY] = calcium_model_reduced([Z, Y], [0], params)
    ax.contour(Z, Y, dZ, levels=[0], linewidths=2, colors='r', label='Z Nullcline')
    ax.contour(Z, Y, dY, levels=[0], linewidths=2, colors='g', label='Y Nullcline')

    # Find the equilibrium point
    steady_state = fsolve(root_finding_problem, [0.5,0.5], args=params)
    print('Location of the steady state: ',steady_state)
    ax.scatter(steady_state[0], steady_state[1], zorder=10, marker='*', s=100, c ='k', label='Steady State')
    ax.legend()

    v,w = get_stability(steady_state, params)


    return 0


def main():

    # Define time period for solution
    t = np.linspace(0,10,10000)
    # Define ICs
    X0 = [1,1,0]

    # Set up the figure for Z and Y
    fig, axes = plt.subplots(2,2)

    # First subplot
    ax1 = axes[0][0]
    sol1 = obtain_sol(X0, t, 2, 2, 4, 0.301, 10,1)
    ax1.plot(t, sol1[:,1], '--', label='Y')
    ax1.plot(t, sol1[:,0], label='Z')
    ax1.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax1.set_xlabel('Time /s'), ax1.set_title(r'(a) $\beta=0.301$ and $k=10$')
    ax1.legend()

    # Second subplot
    ax2 = axes[0][1]
    sol2 = obtain_sol(X0, t, 2, 2, 4, 0.644, 10,1)
    ax2.plot(t, sol2[:,1], '--', label='Y')
    ax2.plot(t, sol2[:,0], label='Z')
    ax2.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax2.set_xlabel('Time /s'), ax2.set_title(r'(b) $\beta=0.644$ and $k=10$')
    ax2.legend()

    # Third subplot
    ax3 = axes[1][0]
    sol3 = obtain_sol(X0, t, 2, 2, 4, 0.301, 6,1)
    ax3.plot(t, sol3[:,1], '--', label='Y')
    ax3.plot(t, sol3[:,0], label='Z')
    ax3.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax3.set_xlabel('Time /s'), ax3.set_title(r'(c) $\beta=0.301$ and $k=6$')
    ax3.legend()

    # Fourth Subplot
    ax4 = axes[1][1]
    sol4 = obtain_sol(X0, t, 2, 2, 4, 0.644, 6,1)
    ax4.plot(t, sol4[:,1], '--', label='Y')
    ax4.plot(t, sol4[:,0], label='Z')
    ax4.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax4.set_xlabel('Time /s'), ax4.set_title(r'(d) $\beta=0.644$ and $k=6$')
    ax4.legend()

    fig.tight_layout()



    # Figure for spontaneous spiking
    fig1, axes = plt.subplots(1,2, figsize=[6.4, 2.4])

    # Low v_0
    ax11 = axes[0]
    sol11 = obtain_sol(X0, t, 2, 2, 4, 0.301, 10, 0.1)
    ax11.plot(t, sol11[:,1], '--', label='Y')
    ax11.plot(t, sol11[:,0], label='Z')
    ax11.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax11.set_xlabel('Time /s'), ax11.set_title(r'(a) $\beta=0.304$ and $v_0 =0.1$')
    ax11.legend()

    # Low beta
    ax12 = axes[1]
    sol12 = obtain_sol(X0, t, 2, 2, 4, 0, 10, 3.5)
    ax12.plot(t, sol12[:,1], '--', label='Y')
    ax12.plot(t, sol12[:,0], label='Z')
    ax12.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax12.set_xlabel('Time /s'), ax12.set_title(r'(b) $\beta=0$ and $v_0 =3.5$')
    ax12.legend()

    fig1.tight_layout()



    # Set up the figure for Z and W*
    fig2, axes = plt.subplots(1,2, figsize=[6.4, 2.4])

    # Low beta
    ax21 = axes[0]
    w_av1 = np.mean(sol1[:,2])
    ax21.plot(t, sol1[:,0], '--', label='Z')
    ax21.plot(t, sol1[:,2], label='$W^*$')
    ax21.plot(t, [w_av1]*len(t), label='Average $W^*$')
    ax21.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax21.set_xlabel('Time /s'), ax21.set_title(r'(a) $\beta=0.301$ and $k=10$')
    ax21.legend()

    # High beta
    ax22 = axes[1]
    w_av2 = np.mean(sol2[:,2])
    ax22.plot(t, sol2[:,0], '--', label='Z')
    ax22.plot(t, sol2[:,2], label='$W^*$')
    ax22.plot(t, [w_av2]*len(t), label='Average $W^*$')
    ax22.set_ylabel(r'$\mathrm{Ca}^{2+}$ Conc.'), ax22.set_xlabel('Time /s'), ax22.set_title(r'(b) $\beta=0.644$ and $k=10$')
    ax22.legend()

    fig2.tight_layout()


    # Save the figures
    fig.savefig('Figures/Time_Domain/TD1.pgf')
    fig1.savefig('Figures/Time_Domain/TD2.pgf')
    fig2.savefig('Figures/Time_Domain/freq_enc.pgf')

    # Generate PP1
    fig_phase1, axes = plt.subplots(2,2)
    phase_plot(sol1[:,0], sol1[:,1], axes[0][0], 2, 2, 4, 0.301, 10, 1)
    phase_plot(sol2[:,0], sol2[:,1], axes[0][1], 2, 2, 4, 0.644, 10, 1)
    phase_plot(sol3[:,0], sol3[:,1], axes[1][0], 2, 2, 4, 0.301, 6, 1)
    phase_plot(sol4[:,0], sol4[:,1], axes[1][1], 2, 2, 4, 0.644, 6, 1)
    fig_phase1.tight_layout()
    axes[0][0].set_title(r'(a) $\beta=0.301$ and $k=10$')
    axes[0][1].set_title(r'(b) $\beta=0.644$ and $k=10$')
    axes[1][0].set_title(r'(c) $\beta=0.301$ and $k=6$')
    axes[1][1].set_title(r'(d) $\beta=0.644$ and $k=6$')
    fig_phase1.savefig('Figures/Phase_Plane/portraits1.pgf')

    # Generate PP2
    fig_phase2, axes = plt.subplots(1,2, figsize=[6.4, 2.4])
    phase_plot(sol11[:,0], sol11[:,1], axes[0], 2, 2, 4, 0.301, 10, 0.1)
    phase_plot(sol12[:,0], sol12[:,1], axes[1], 2, 2, 4, 0, 10, 3.5)
    fig_phase2.tight_layout()
    axes[0].set_title(r'(a) $\beta=0.301$ and $v_0 = 0.1$')
    axes[1].set_title(r'(b) $\beta=0.644$ and $v_0 = 3.5$')
    fig_phase2.savefig('Figures/Phase_Plane/portraits2.pgf')


    # Show the figures
    plt.show()




if __name__ == '__main__':
    main()