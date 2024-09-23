from algorithm import PSO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def demo_func(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return x[:,0] ** 2 + (x[:,1] - 0.05) ** 2

p = 1000
d = 2
g = 150
iters = 20
x_max = 1*np.ones(d)
x_min = -1*np.ones(d)
optimizer = PSO(funct=demo_func, num_dim=d, num_particle=p, max_iter=iters, x_max=x_max, x_min=x_min,c1=0.5, c2=0.5)
optimizer.update()
optimizer.plot_curve()

X_list = optimizer.X_history
V_list = optimizer.V_history
fig, ax = plt.subplots(1, 1)
ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
Z_grid = np.zeros((40,40))
for i in range(Z_grid.shape[1]):
    Z_grid[:,i] = demo_func(np.vstack([X_grid[:,i],Y_grid[:,i]]).T)
ax.contour(X_grid, Y_grid, Z_grid, 20)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.ion()
p = plt.show()
def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line
ani = FuncAnimation(fig, update_scatter, blit=False, interval=25, frames=500)
ani.save('pso.gif', writer='pillow', fps=20)
