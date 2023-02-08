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
pio.renderers.default = "browser"

#vispy.app.use_app("pyqt5")


g_1_0 = -29404.8 * 1e-9
g_1_1 = -1450.9 * 1e-9
h_1_1 = 4652.5 * 1e-9


# Dipole component of Earth's magnetic field in spherical coords
def B_dip(r, theta, phi, R=6400*1e3):
    global g_1_0, g_1_1, h_1_1
    B_r = 2*(R/r)**3*(g_1_0*np.cos(theta)+(g_1_1*np.cos(phi)+h_1_1*np.sin(phi))*np.sin(theta))
    B_theta = -(R/r)**3*(-g_1_0*np.sin(theta)+(g_1_1*np.cos(phi)+h_1_1*np.sin(phi))*np.cos(theta))
    B_phi = -(R/r)**3*(-g_1_1*np.sin(phi)+h_1_1*np.cos(phi))
    return np.array([[B_r, B_theta, B_phi]])


# Spherical -> Cartesian
def transform(theta, phi):
    C = np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],
                  [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
                  [-np.sin(theta), np.cos(phi), 0]])
    return C


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
    theta = np.arccos(z / r)
    phi = np.arctan(y / (x + 1e-3)) if x > 0 else np.arctan(y / (x + 1e-3)) + np.pi
    B_spher = B_dip(r, theta, phi)
    C = transform(theta, phi)
    B = B_spher @ C
    return B[0]


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
    [phi, theta] = np.mgrid[0:2 * np.pi:12j, 0:np.pi:12j]
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
    t = np.linspace(0, t_0, 10000)
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


def magnetic_field(r, theta, phi):
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
                B_spher = B_dip(r[i], theta[j], phi[k])
                C = transform(theta[j], phi[k])
                B_xyz = B_spher @ C
                B[count] = B_xyz
                count += 1

    return B


def magn_lines(start_point, step=1e5):
    r, theta, phi = start_point
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    line = []
    line.append(np.array([x, y, z]))
    B_start = B_xyz(x, y, z)
    approx_steps = int(np.pi * r)
    i = 1
    while True:
        new_point = line[-1] + step * B_start / np.linalg.norm(B_start)
        line.append(new_point)
        i += 1
        B_start = B_xyz(*new_point)
        if line[-1][0] < 1e-2 and line[-1][1] < 1e-2:
            break
    return line


start = np.array([6400*1e3*1.5, 0, 0])
line = np.array(magn_lines(start))
mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], tube_radius=None)
plot_sphere()
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

#plot_xyz(t, x, y, z)
#fig = plt.figure(figsize=(16, 9))
#ax = fig.add_subplot(111, projection='3d')
#plot_sphere()
#ax.plot(x, y, z)
theta = np.array([x * np.pi / 40 for x in range(40)])
phi = np.array([x * np.pi / 3 for x in range(6)])
r = np.array([6400 * 1e3 * (1 + x) for x in range(1, 3)])
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


#fig = plt.figure(figsize=(16, 9))
#ax = fig.add_subplot(111, projection='3d')
B = magnetic_field(r, theta, phi)



#test_flow()
#mlab.axes()
#mlab.show()
#ax.quiver(x_, y_, z_, B[:, 0]*1e11, B[:, 1]*1e11, B[:, 2]*1e11)
#plt.show()
#plot_B(B=B, r=r, theta=theta, phi=phi, show=True, save=False)
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




#initial_conditions = [proton_1]
#particles_trajectories(initial_conditions, save=False, t_0=100, traj_plot=False)
