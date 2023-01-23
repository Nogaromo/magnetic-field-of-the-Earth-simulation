import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits import mplot3d


g_1_0 = -29404.8 * 1e-9
g_1_1 = -1450.9 * 1e-9
h_1_1 = 4652.5 * 1e-9
c = 3 * 1e8


def B_dip(r, theta, phi, R=6400*1e3):
    global g_1_0, g_1_1, h_1_1
    B_r = 2*(R/r)**3*(g_1_0*np.cos(theta)+(g_1_1*np.cos(phi)+h_1_1*np.sin(phi))*np.sin(theta))
    B_theta = -(R/r)**3*(-g_1_0*np.sin(theta)+(g_1_1*np.cos(phi)+h_1_1*np.sin(phi))*np.cos(theta))
    B_phi = -(R/r)**3*(-g_1_1*np.sin(phi)+h_1_1*np.cos(phi))
    return np.array([[B_r, B_theta, B_phi]])


def transform(theta, phi):
    C = np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],
                  [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                  [-np.sin(theta), np.cos(phi), 0]])
    return C


def plot_B(B):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = 0
    for i in tqdm(range(n)):
        for j in range(n):
            for k in range(n):
                B_curr = B[c]
                x = r[i] * np.sin(theta[j]) * np.cos(phi[k])
                y = r[i] * np.sin(theta[j]) * np.sin(phi[k])
                z = r[i] * np.cos(theta[j])
                c += 1
                ax.quiver(x, y, z, 1e11 * B_curr[0], 1e11 * B_curr[1], 1e11 * B_curr[2], color='r')
    plt.show()


def B_xyz(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)
    B_spher = B_dip(r, theta, phi)
    C = transform(theta, phi)
    B = B_spher @ C
    #print(B_spher.shape, C.shape)
    #print(B[0])
    return B[0]
    

def eqn(y, t, e=1.6e-19, m=1.67e-27):
    gamma = e / m
    y_1, y_2, y_3, y_4, y_5, y_6 = y
    B_x, B_y, B_z = B_xyz(y_1, y_3, y_5)
    dydt = [y_2, gamma*(y_4 * B_z - y_6 * B_y), y_4, -gamma*(y_2 * B_z - y_6 * B_x), y_6, gamma*(y_2 * B_y - y_4 * B_x)]
    return dydt


def plot_xyz(t, x, y, z):
    plt.figure(dpi=200)
    plt.grid()
    plt.plot(t, x, label='x(t)')
    plt.plot(t, y, label='y(t)')
    plt.plot(t, z, label='z(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.show()


def plot_sphere(r=6400 * 1e3):
    plt.rcParams["figure.autolayout"] = True
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    ax.plot_surface(r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v), cmap=plt.cm.YlGnBu_r)
    #plt.show()


x0 = 6400 * 1e3 * 3
y0 = 6400 * 1e3 * 3
z0 = 6400 * 1e3 * 3
v_0_x = 1.38 * 1e7 / np.sqrt(3)
v_0_y = 1.38 * 1e7 / np.sqrt(3)
v_0_z = 1.38 * 1e7 / np.sqrt(3)
initial_cond = [x0, v_0_x, y0, v_0_y, z0, v_0_z]
t = np.linspace(0, 1000, 10000)
sol = odeint(eqn, initial_cond, t)

x = sol[:, 0]
y = sol[:, 2]
z = sol[:, 4]
v_x = sol[:, 1]
v_y = sol[:, 3]
v_z = sol[:, 5]
plot_xyz(t, x, y, z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_sphere()
ax.plot(x, y, z)
plt.show()
