import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
class PSO():
    def __init__(self, funct, num_dim,x_max, x_min, num_particle=20, max_iter=500,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.funct = funct
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.x_max = x_max
        self.x_min = x_min
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self._iter = 1
        self.global_best_curve = np.zeros(self.max_iter)
        self.X = np.random.uniform(low=self.x_min, high=self.x_max, size=[self.num_particle, self.num_dim])#初始化粒子
        self.V = np.zeros(shape=[self.num_particle, self.num_dim])#初始化速率
        self.v_max = self.k*(self.x_max-self.x_min)/2
        self.individual_best_solution = self.X.copy()
        self.individual_best_value = self.funct(self.X)
        self.global_best_solution = self.individual_best_solution[self.individual_best_value.argmin()]
        self.global_best_value = self.individual_best_value.min()
        self.X_history = []
        self.V_history = []
    def update(self):
        while self._iter <= self.max_iter:
            self.X_history.append(self.X.copy())
            self.V_history.append(self.V.copy())
            R1 = np.random.uniform(size=(self.num_particle, self.num_dim))
            R2 = np.random.uniform(size=(self.num_particle, self.num_dim))
            w = self.w_max - self._iter*(self.w_max-self.w_min)/self.max_iter
            for i in range(self.num_particle):
                self.V[i, :] = w*self.V[i, :] + self.c1*(self.individual_best_solution[i,:] - self.X[i,:])*R1[i,:] + self.c2*(self.global_best_solution - self.X[i,:])*R2[i,:]
                self.V[i, self.v_max < self.V[i, :]] = self.v_max[self.v_max < self.V[i, :]]
                self.V[i, -self.v_max > self.V[i, :]] = -self.v_max[-self.v_max > self.V[i, :]]
                self.X[i, :] = self.X[i, :] + self.V[i, :]
                self.X[i, self.x_max < self.X[i, :]] = self.x_max[self.x_max < self.X[i, :]]
                self.X[i, self.x_min > self.X[i, :]] = self.x_min[self.x_min > self.X[i, :]]

                score = self.funct(self.X[i, :])
                if score<self.individual_best_value[i]:
                    self.individual_best_value[i] = score.copy()
                    self.individual_best_solution[i, :] = self.X[i, :].copy()
                    if score<self.global_best_value:
                        self.global_best_value = score.copy()
                        self.global_best_solution = self.X[i, :].copy()
            self.global_best_curve[self._iter-1] = self.global_best_value.copy()
            self._iter += 1
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.global_best_curve[-1], 3))+']')
        plt.plot(self.global_best_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()