import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
from vispy import plot as vp
import vispy
from vispy import scene
from tqdm import tqdm
import vispy.io as io
import imageio
import plotly.graph_objects as go
from pyface.qt import QtGui, QtCore
from mayavi.mlab import quiver3d
from mayavi import mlab
import plotly.io as pio
from scipy.special import sph_harm
import pyvista as pv


#vispy.app.use_app("pyqt5")


g_1_0 = -29404.8 * 1e-9 * 0
g_1_1 = -1450.9 * 1e-9 * 0
h_1_1 = 4652.5 * 1e-9 * 0
g_2_0 = -2499.6 * 1e-9
g_2_1 = 2982.0 * 1e-9 * 0
h_2_1 = -2991.6 * 1e-9 * 0
g_2_2 = 1677.0 * 1e-9 * 0
h_2_2 = -734.6 * 1e-9 * 0
Re = 6400 * 1e3
Rm = 16 * Re
b = 5 * Re


# Dipole component of Earth's magnetic field in spherical coords
def B_dip(r, theta, phi, R=6400*1e3):
    global g_1_0, g_1_1, h_1_1
    B_r = 2*(R/r)**3*(g_1_0*np.cos(theta)+(g_1_1*np.cos(phi)+h_1_1*np.sin(phi))*np.sin(theta))
    B_theta = -(R/r)**3*(-g_1_0*np.sin(theta)+(g_1_1*np.cos(phi)+h_1_1*np.sin(phi))*np.cos(theta))
    B_phi = -(R/r)**3*(-g_1_1*np.sin(phi)+h_1_1*np.cos(phi))
    return np.array([B_r, B_theta, B_phi])


def B_qp(r, theta, phi, R=6400*1e3):
    global g_1_0, g_1_1, h_1_1, g_2_0, g_2_1, h_2_1, g_2_2, h_2_2
    B_r = 3*(R/r)**4*(g_2_0*(3*np.cos(theta)**2-1)/2+(g_2_1*np.cos(phi)+h_2_1*np.sin(phi))*np.sqrt(3)/2*np.sin(2*theta)+
                      (g_2_2*np.cos(2*phi)+h_2_2*np.sin(phi))*np.sqrt(3)/2*np.sin(theta)**2)
    B_theta = -(R/r)**4*(-g_2_0*3/2*np.sin(2*theta)+(g_2_1*np.cos(phi)+h_2_1*np.sin(phi))*np.sqrt(3)*np.cos(2*theta)+
                         (g_2_2*np.cos(2*phi)+h_2_2*np.sin(2*phi))*np.sqrt(3)/2*np.sin(2*theta))
    B_phi = np.sqrt(3)*(R/r)**4*((g_2_1*np.sin(phi)-h_2_1*np.cos(phi))*np.cos(theta)+(g_2_2*np.sin(2*phi)-h_2_2*np.cos(2*phi))*np.sin(theta))
    return np.array([B_r, B_theta, B_phi])


def c_n_k(n, k):
    #global g_1_0, g_1_1, h_1_1, g_2_0, g_2_1, h_2_1, g_2_2, h_2_2
    g_1_0 = -29404.8 * 1e-9 * 0
    g_1_1 = -1450.9 * 1e-9 * 0
    h_1_1 = 4652.5 * 1e-9 * 0
    g_2_0 = -2499.6 * 1e-9
    g_2_1 = 2982.0 * 1e-9 * 0
    h_2_1 = -2991.6 * 1e-9 * 0
    g_2_2 = 1677.0 * 1e-9 * 0
    h_2_2 = -734.6 * 1e-9 * 0
    if k == 0:
        if n == 1:
            return g_1_0
        if n == 2:
            return g_2_0
    if n == 1:
        return np.sqrt(g_1_1**2+h_1_1**2)
    if n == 2:
        if k == 1:
            return np.sqrt(g_2_1**2+h_2_1**2)
        if k == 2:
            return np.sqrt(g_2_2**2+h_2_2**2)


def y_n_k(theta, phi, n, k):
    if k > n:
        return 0
    else:
        return sph_harm(k, n, phi, theta)


def B_cf(r, theta, phi, R=6400*1e3, N=2):
    global g_1_0, g_1_1, h_1_1, g_2_0, g_2_1, h_2_1, g_2_2, h_2_2, Rm
    B_r = 0
    B_theta = 0
    B_phi = 0
    for n in range(1, N+1):
        for k in range(0, n+1):
            c_n_k_coeff = c_n_k(n, k)
            B_r -= np.real(c_n_k_coeff * y_n_k(phi, theta, n, k) * (r/Rm)**(n-1))
            B_theta -= np.real(c_n_k_coeff/n * (r/Rm)**n * (k*1/np.tan(theta) * y_n_k(phi, theta, n, k) +
                                              np.sqrt((n-k)*(n+k+1))*y_n_k(phi, theta, n, k+1)))
            B_phi -= np.real(c_n_k_coeff/n * (r/Rm)**n * 1j * k * y_n_k(phi, theta, n, k))
    if 0 < phi < np.pi:
        return np.array([B_r, Rm/r * B_theta, Rm/(r*np.sin(theta)) * B_phi])
    else:
        return np.array([0, 0, 0])


def B_cf_(r, theta, phi):
    global g_2_0, Rm
    B_r = -r / Rm * g_2_0 * np.sqrt(5/(4*np.pi))*(3/2*np.cos(theta)**2-1/2)
    B_theta = np.sqrt(6) * g_2_0/2 * r / Rm * np.sqrt(15/(8*np.pi))*np.sin(theta)*np.cos(theta)*np.cos(phi)
    if 0 < phi < np.pi:
        return np.array([B_r, B_theta, 0])
    else:
        return np.array([0, 0, 0])


# Spherical -> Cartesian
def transform(theta, phi):
    C = np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],
                  [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                  [-np.sin(theta), np.cos(phi), 0]])
    return C


def M_system_to_Cartesian(r, theta, phi):
    global b
    if r*np.sin(theta)*np.cos(phi)+b > 0:
        theta_m = np.arctan(-np.cos(theta)/(np.abs(np.cos(theta)))*r*np.sqrt(np.cos(theta)**2+np.sin(theta)**2*np.cos(phi)**2)/(r*np.sin(theta)*np.cos(phi)+b))
    else:
        theta_m = np.arctan(-np.cos(theta)/(np.abs(np.cos(theta)))*r * np.sqrt(np.cos(theta) ** 2 + np.sin(theta) ** 2 * np.cos(phi) ** 2) / (
                    r * np.sin(theta) * np.cos(phi) + b)) + np.pi
    if np.cos(theta) > 0:
        phi_m = np.arctan(np.tan(theta)*np.sin(phi))
    else:
        phi_m = np.arctan(np.tan(theta) * np.sin(phi)) + np.pi
    C_M = np.array([[np.sin(theta_m), -np.cos(phi_m), 0],
                    [np.cos(theta_m)*np.cos(phi_m), np.cos(theta_m)*np.sin(phi_m), -np.sin(theta_m)],
                    [-np.sin(theta_m)*np.cos(phi_m), -np.sin(theta_m)*np.sin(phi_m), -np.cos(theta_m)]])
    return C_M


# Plots lines of the magnetic field
def plot_B(B, r, theta, phi, show=True, save=False):

    c = 0
    x = np.zeros(r.shape[0] * theta.shape[0] * phi.shape[0])
    y = np.zeros(r.shape[0] * theta.shape[0] * phi.shape[0])
    z = np.zeros(r.shape[0] * theta.shape[0] * phi.shape[0])
    for i in tqdm(range(r.shape[0])):
        for j in range(theta.shape[0]):
            for k in range(phi.shape[0]):
                x[c] = r[i] * np.sin(theta[j]) * np.cos(phi[k])
                y[c] = r[i] * np.sin(theta[j]) * np.sin(phi[k])
                z[c] = r[i] * np.cos(theta[j])
                c += 1
    if show:
        quiver3d(x, y, z, B[:, 0], B[:, 1], B[:, 2])
        plot_sphere()
        #mlab.show()
    #if save:
        #plt.savefig('magnetic_lines.png', dpi=600)


# Returns magnetic field in Cartesian coordinate system
def B_xyz(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    r_cf = np.sqrt((x+b)**2+y**2+z**2)
    #r_cf = r
    theta = np.arccos(z / r)
    phi = np.arctan(y / x) if x > 0 else np.arctan(y / x) + np.pi
    B_spher_earth = B_dip(r, theta, phi) + B_qp(r, theta, phi)
    B_cf_M = B_cf_(r_cf, theta, phi)
    C_M = M_system_to_Cartesian(r, theta, phi)
    C = transform(theta, phi)
    B_cf1 = B_cf_M @ C_M
    B_earth = B_spher_earth @ C
    B = B_earth + B_cf1
    return B


# System of differential equations
def eqn(y, t, e=1.6e-19, m=1.67e-27):
    gamma = e / m
    y_1, y_2, y_3, y_4, y_5, y_6 = y
    B_x, B_y, B_z = B_xyz(y_1, y_3, y_5)
    dydt = [y_2, gamma*(y_4 * B_z - y_6 * B_y), y_4, -gamma*(y_2 * B_z - y_6 * B_x), y_6, gamma*(y_2 * B_y - y_4 * B_x)]
    return dydt


# Creates x(t), y(t) and z(t) graphs
def plot_xyz(t, x, y, z):
    plt.figure(dpi=200)
    plt.grid()
    plt.plot(t, x, label='x(t)')
    plt.plot(t, y, label='y(t)')
    plt.plot(t, z, label='z(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.show()


# Creates sphere (mayavi)
def plot_sphere(r=6400 * 1e3, x_0=0, y_0=0, z_0=0):
    [phi, theta] = np.mgrid[0:2 * np.pi:30j, 0:np.pi:30j]
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    mlab.mesh(x + x_0, y + y_0, z + z_0, colormap='gist_earth')


# Creates sphere and proton trajectory plot with OpenGL
def plot_with_gpu(solutions, canvas=None, magnetic_field=None, save=False):
    view = canvas.central_widget.add_view()
    view.camera = 'arcball'
    axes = scene.visuals.XYZAxis()
    grid = canvas.central_widget.add_grid()
    t = 1
    for solution in solutions:
        x = solution[:, 0]
        y = solution[:, 2]
        z = solution[:, 4]
        r, g, b = np.random.randint(1, [256, 256, 256])
        color = '%02x%02x%02x' % (r, g, b)
        plot = scene.Line(np.array([x, y, z]).T, parent=view.scene, color=f'#{color}')
        t1 = scene.visuals.Text(f'particle №{t}', parent=canvas.scene, color=f'#{color}', pos=(50, (t + 1) * 20))
        t += 1
    mf = scene.Line(magnetic_field, parent=view.scene, color='black')
    sphere1 = scene.visuals.Sphere(radius=6400 * 1e3, method='ico', parent=view.scene,
                                   edge_color='black')
    axes = scene.visuals.XYZAxis(parent=view.scene)
    view.camera.set_range(x=[-6400 * 1e4 / 1.5, 6400 * 1e4 / 1.5])
    text1 = scene.visuals.Text("Proton trajectory in Earths's dipole magnetic field", pos=(230, 15), font_size=14,
                               color='black', parent=canvas.scene)

    if save:
        #img = canvas.render()
        #io.write_png('protons trajectories.png', img)
        axis = [0, 0, 1]
        n_steps = 72
        step_angle = 5.
        writer = imageio.get_writer('animation.gif')
        for i in range(n_steps * 2):
            im = canvas.render(alpha=True)
            writer.append_data(im)
            if i >= n_steps:
                view.camera.transform.rotate(step_angle, axis)
            else:
                view.camera.transform.rotate(-step_angle, axis)
        writer.close()


# Plots particles trajectories
def particles_trajectories(initial_conditions, t_0=100, traj_plot=True, save=False, axes=True, show=True):
    solutions = []
    t = np.linspace(0, t_0, t_0*100)
    for initial_cond in initial_conditions:
        sol = odeint(eqn, initial_cond, t)
        solutions.append(sol)

    for solution in solutions:
        x = np.array(solution[:, 0])
        y = np.array(solution[:, 2])
        z = np.array(solution[:, 4])
        n1, n2, n3 = np.random.rand(3)
        if traj_plot:
            mlab.plot3d(x, y, z, tube_radius=None, color=(n1, n2, n3))
    if axes:
        mlab.axes(line_width=2)
    if save:
        mlab.savefig(filename='trajectories.png', size=(1920, 1080))
    if show:
        mlab.show()


        #canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                                   #size=(1280, 720), show=True)
        #plot_with_gpu(solutions, canvas=canvas, magnetic_field=None, save=save)
        #canvas.app.run()


def magnetic_field(r, theta, phi, r_cf=None):
    if r_cf is None:
        r_cf = r
    try:
        n_r = r.shape[0]
        n_theta = theta.shape[0]
        n_phi = phi.shape[0]
    except IndexError:
        n_r = 1
        n_theta = 1
        n_phi = 1
        r = [r]
        theta = [theta]
        phi = [phi]
    n = n_r * n_theta * n_phi
    B = np.zeros((n, 3))
    count = 0
    for i in tqdm(range(n_r)):
        for j in range(n_theta):
            for k in range(n_phi):
                B_spher = B_dip(r[i], theta[j], phi[k]) + B_qp(r[i], theta[j], phi[k])
                B_cf_M = B_cf_(r_cf[i], theta[j], phi[k])
                C = transform(theta[j], phi[k])
                C_M = M_system_to_Cartesian(r[i], theta[j], phi[k])
                B_cf_xyz = B_cf_M @ C_M
                B_xyz_earth = B_spher @ C
                B_xyz = B_xyz_earth + B_cf_xyz
                B[count] = B_xyz
                count += 1

    return B


def magn_lines(start_points, step=1e5, R_bound = 6400*1e3*11):
    global Re
    lines = []
    for point in start_points:
        r, theta, phi = point
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        line = []
        b = [0, 0]
        line.append(np.array([x, y, z]))
        B_start = B_xyz(x, y, z)
        x = np.sign(theta - np.pi / 2)
        while Re <= r <= R_bound:
            new_point = line[-1] + x * step * B_start / np.linalg.norm(B_start)
            r = np.linalg.norm(new_point)
            line.append(new_point)
            B_start = B_xyz(*new_point)
            b.append(np.linalg.norm(B_start))

            if b[-1] == b[-3]:
                print(1)
                break
        lines.append(line)
    return lines


def plot_magnetic_field_lines(theta, phi):
    start_points = []
    for i in range(theta.shape[0]):
        for j in range(phi.shape[0]):
            theta_ = theta[i]
            phi_ = phi[j]
            point = np.array([6400 * 1e3, theta_, phi_])
            start_points.append(point)
    lines = magn_lines(start_points)
    for line in lines:
        line = np.array(line)
        if line.shape[0] == 2:
            pass
        else:
            mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], tube_radius=None, color=(0.1, 0.3, 0.5))
    plot_sphere()


#theta_1 = [np.pi/12, np.pi/6, np.pi/3, np.pi/4]
#theta_2 = [np.pi/2 + np.pi/48, np.pi/2 + np.pi/24, np.pi/2 + np.pi/12, np.pi/2 + np.pi/6]
theta_1 = [np.pi / x for x in range(3, 13, 1)]
theta_2 = [np.pi/2 + np.pi / x for x in range(3, 25, 3)]
theta = np.concatenate((theta_1, theta_2), axis=None)
phi = np.array([x * np.pi / 6 for x in range(13)])
plot_magnetic_field_lines(theta, phi)
mlab.plot3d([0, 0], [0, 0], [0, 6400*1e3*10], tube_radius=None, color=(0.7, 0.3, 0.5))
mlab.plot3d([0, 0], [0, 6400*1e3*10], [0, 0], tube_radius=None, color=(0.7, 0.3, 0.5))
mlab.plot3d([0, 6400*1e3*10], [0, 0], [0, 0], tube_radius=None, color=(0.7, 0.3, 0.5))
mlab.view(focalpoint=[0, 0, 0])
mlab.show()

x0 = 6400 * 1e3 * 3
y0 = 6400 * 1e3 * 3
z0 = 6400 * 1e3 * 3
v_0_x = 1.38 * 1e7 / np.sqrt(3)
v_0_y = 1.38 * 1e7 / np.sqrt(3)
v_0_z = 1.38 * 1e7 / np.sqrt(3)
initial_cond = [x0, v_0_x, y0, v_0_y, z0, v_0_z]
t = np.linspace(0, 10, 1000)
sol = odeint(eqn, initial_cond, t)

x = sol[:, 0]
y = sol[:, 2]
z = sol[:, 4]
v_x = sol[:, 1]
v_y = sol[:, 3]
v_z = sol[:, 5]


theta = np.array([x * np.pi / 40 for x in range(40)])
phi = np.array([x * np.pi / 3 for x in range(6)])
r = np.array([6400 * 1e3 * (1 + 0.5*x) for x in range(1, 5)])
r_cf = r + b
c = 0
x_ = np.zeros(r.shape[0] * theta.shape[0] * phi.shape[0])
y_ = np.zeros(r.shape[0] * theta.shape[0] * phi.shape[0])
z_ = np.zeros(r.shape[0] * theta.shape[0] * phi.shape[0])
for i in tqdm(range(r.shape[0])):
    for j in range(theta.shape[0]):
        for k in range(phi.shape[0]):
            x_[c] = r[i] * np.sin(theta[j]) * np.cos(phi[k])
            y_[c] = r[i] * np.sin(theta[j]) * np.sin(phi[k])
            z_[c] = r[i] * np.cos(theta[j])
            c += 1

#B = magnetic_field(r, theta, phi)
#mlab.quiver3d(x_, y_, z_, B[:, 0], B[:, 1], B[:, 2])
#plot_sphere()
#mlab.show()
x0 = 6400 * 1e3 * 3
y0 = 6400 * 1e3 * 3
z0 = 6400 * 1e3 * 3
v_0_x = 1.38 * 1e7 / np.sqrt(3)
v_0_y = 1.38 * 1e7 / np.sqrt(3)
v_0_z = 1.38 * 1e7 / np.sqrt(3)
proton_1 = [x0, v_0_x, y0, v_0_y, z0, v_0_z]
proton_2 = [x0 / 2, v_0_x, y0 / 2, v_0_y, z0 / 2, v_0_z]
proton_3 = [-8000 * 1e3 * np.sqrt(2), 0, -8000 * 1e3, 0, z0, 1.38 * 1e7]

initial_conditions = [proton_1, proton_2, proton_3]
#les_trajectories(initial_conditions, save=True, t_0=100, traj_plot=True, axes=False)


