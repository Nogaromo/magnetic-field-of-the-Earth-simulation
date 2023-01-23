import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits import mplot3d

g_1_0 = -29404.8 * 1e-9
g_1_1 = -1450.9 * 1e-9
h_1_1 = 4652.5 * 1e-9


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
    return B_spher @ C
    

#n = 10
#theta = [x * np.pi / n for x in range(n)]
#phi = [x * 2 * np.pi / n for x in range(n)]
#r = [6400*1e3*(1+x/5) for x in range(n)]
#B = np.zeros((n**3, 3))
#count = 0
#for i in tqdm(range(n)):
    #for j in range(n):
        #for k in range(n):
            #B_spher = B_dip(r[i], theta[j], phi[k])
            #C = transform(theta[j], phi[k])
            #B_xyz = B_spher @ C
            #B[count] = B_xyz
            #count += 1
            
#plot_B(B)
