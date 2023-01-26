import sys
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
vispy.app.use_app("pyqt6")


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
    for i in tqdm(range(r.shape[0])):
        for j in range(theta.shape[0]):
            for k in range(phi.shape[0]):
                B_curr = B[c]
                x = r[i] * np.sin(theta[j]) * np.cos(phi[k])
                y = r[i] * np.sin(theta[j]) * np.sin(phi[k])
                z = r[i] * np.cos(theta[j])
                c += 1
                ax.quiver(x, y, z, 1e11 * B_curr[0], 1e11 * B_curr[1], 1e11 * B_curr[2], color='r', length=100)
    if show:
        plt.show()
    if save:
        plt.savefig('magnetic_lines.png', dpi=600)


# Returns magnetic field in Cartesian coordinate system
def B_xyz(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x) if x > 0 else np.arctan(y / x) + np.pi
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


# Creates sphere (matplotlib)
def plot_sphere(r=6400 * 1e3):
    plt.rcParams["figure.autolayout"] = True
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    ax.plot_surface(r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v), cmap=plt.cm.YlGnBu_r)


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
def particles_trajectories(initial_conditions, plot=True, save=False):
    solutions = []
    t = np.linspace(0, 100, 100000)
    for initial_cond in initial_conditions:
        sol = odeint(eqn, initial_cond, t)
        solutions.append(sol)
    if plot:
        canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                                   size=(1280, 720), show=True)
        plot_with_gpu(solutions, canvas=canvas, magnetic_field=None, save=save)
        canvas.app.run()


def magnetic_field(r, theta, phi):
    n = r.shape[0] * theta.shape[0] * phi.shape[0]
    B = np.zeros((n, 3))
    count = 0
    for i in tqdm(range(r.shape[0])):
        for j in range(theta.shape[0]):
            for k in range(phi.shape[0]):
                B_spher = B_dip(r[i], theta[j], phi[k])
                C = transform(theta[j], phi[k])
                B_xyz = B_spher @ C
                B[count] = B_xyz
                count += 1

    return B


x0 = 6400 * 1e3 * 3
y0 = 6400 * 1e3 * 3
z0 = 6400 * 1e3 * 3
v_0_x = 1.38 * 1e7 / np.sqrt(3)
v_0_y = 1.38 * 1e7 / np.sqrt(3)
v_0_z = 1.38 * 1e7 / np.sqrt(3)
initial_cond = [x0, v_0_x, y0, v_0_y, z0, v_0_z]
t = np.linspace(0, 100, 10000)
sol = odeint(eqn, initial_cond, t)

x = sol[:, 0]
y = sol[:, 2]
z = sol[:, 4]
v_x = sol[:, 1]
v_y = sol[:, 3]
v_z = sol[:, 5]

#plot_xyz(t, x, y, z)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
plot_sphere()
#ax.plot(x, y, z)
theta = np.array([x * np.pi / 20 for x in range(20)])
phi = np.array([x * np.pi / 5 for x in range(10)])
r = np.array([6400 * 1e3 * (2 + x) for x in range(2, 4)])
plot_B(B=magnetic_field(r, theta, phi), r=r, theta=theta, phi=phi, show=False, save=False)

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
particles_trajectories(initial_conditions, save=True)
