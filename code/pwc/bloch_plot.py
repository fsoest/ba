import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from qutip import Bloch
from numpy import linspace, outer, arctan2, arccos, pi, sin, cos, ones, zeros, size, empty
from N2_analyt import E
from helpers import bloch_to_vec

X = np.load('train_data/X_N_2_50000_dt_3.npy')
y = np.load('train_data/y_N_2_50000_dt_3.npy')

#E_calc = E(y[:,], X[:, 0], X[:, 2], 3)
min = 1353#E_calc.argmin()
y_min = y[min]
X_min = X[min]

def hm(theta, phi):
    #t_t, t_prime, p_t, p_prime
    trans = np.array([[y_min[0], theta, y_min[2], phi]])
    return -E(trans, X_min[0], X_min[2], 3)


class HeatmapBloch(Bloch):
    def __init__(self, heatmap_function, cmap):
        """
        Heatmap function of theta, phi
        """
        super().__init__()
        self.heatmap_function = heatmap_function
        self.smap = cm.ScalarMappable(cmap=cmap)

    def plot_back(self):
        # back half of sphere
        u = linspace(0, pi, 100)
        v = linspace(0, pi, 100)
        x = outer(cos(u), sin(v))
        y = outer(sin(u), sin(v))
        z = outer(ones(size(u)), cos(v))

        hm = zeros(x.shape)

        for i in range(len(x)):
            for j in range(len(y)):
                hm[i, j] = self.heatmap_function(arctan2(y[i, j], x[i, j]), arccos(z[i,j]))

        colours = self.smap.to_rgba(hm)

        self.axes.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colours, linewidth=0, alpha=self.sphere_alpha)
        # wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5, color=self.frame_color, alpha=self.frame_alpha)
        # equator
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='z', lw=self.frame_width, color=self.frame_color)
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='x', lw=self.frame_width, color=self.frame_color)

    def plot_front(self):
        # front half of sphere
        u = linspace(-pi, 0, 100)
        v = linspace(0, pi, 100)
        x = outer(cos(u), sin(v))
        y = outer(sin(u), sin(v))
        z = outer(ones(size(u)), cos(v))

        hm = zeros(x.shape)

        for i in range(len(x)):
            for j in range(len(y)):
                hm[i, j] = self.heatmap_function(np.arctan2(y[i, j], x[i, j]), np.arccos(z[i,j]))

        colours = self.smap.to_rgba(hm)

        self.axes.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colours, linewidth=0, alpha=self.sphere_alpha)
        #wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5, color=self.frame_color, alpha=self.frame_alpha)
        # equator
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='z', lw=self.frame_width, color=self.frame_color)
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='x', lw=self.frame_width, color=self.frame_color)

b = HeatmapBloch(hm, 'bwr')
b.sphere_alpha=0.2
x = np.array([np.sin(y_min[1]) * np.cos(y_min[3])])
y = np.array([np.sin(y_min[1]) * np.sin(y_min[3])])
z = np.array([np.cos(y_min[1])])
b.add_points([x, y, z])
b.point_color = ['g']
b.show()
b.fig.colorbar(b.smap, ax=b.axes, shrink=0.5)
plt.show()

hm(y_min[1], y_min[3]-np.pi/10)
hm(y_min[1], y_min[3])
hm(np.pi/2, 0)
# %%
the = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
res = np.zeros((100, 100))
for i,t in enumerate(the):
    for j, p in enumerate(phi):
        res[i, j] = hm(t, p)

res.max()
res.min()
