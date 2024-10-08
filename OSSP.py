import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, num_jobs, num_machines):
        self.position = np.random.permutation(num_jobs * num_machines)
        self.velocity = np.zeros(num_jobs * num_machines)
        self.best_position = self.position.copy()
        self.best_makespan = float('inf')


def makespan(schedule, processing_times, machines, num_jobs, num_machines):
    machine_end_time = [0] * num_machines
    job_end_time = [0] * num_jobs

    for task in schedule:
        job = task // num_machines
        machine = machines[job][task % num_machines] - 1
        start_time = max(machine_end_time[machine], job_end_time[job])
        machine_end_time[machine] = start_time + processing_times[job][task % num_machines]
        job_end_time[job] = machine_end_time[machine]

    return max(machine_end_time)


def update_velocity(particle, global_best_position, inertia_weight, cognitive, social):
    r1, r2 = np.random.rand(), np.random.rand()
    cognitive_velocity = cognitive * r1 * (particle.best_position - particle.position)
    social_velocity = social * r2 * (global_best_position - particle.position)
    particle.velocity = inertia_weight * particle.velocity + cognitive_velocity + social_velocity


def update_position(particle):
    num_elements = len(particle.position)

    for i in range(num_elements):
        if abs(particle.velocity[i]) > np.random.rand():
            swap_index = np.random.randint(0, num_elements)
            particle.position[i], particle.position[swap_index] = particle.position[swap_index], particle.position[i]

    particle.position = np.clip(particle.position, 0, num_elements - 1).astype(int)


def pso(processing_times, machines, num_jobs, num_machines, num_particles=150, max_iter=200, inertia_weight=0.9,
        cognitive=2, social=2):
    swarm = [Particle(num_jobs, num_machines) for _ in range(num_particles)]
    global_best_position = None
    global_best_makespan = float('inf')
    loss_history = []

    for iteration in range(max_iter):
        for particle in swarm:
            print(particle.position)
            current_makespan = makespan(particle.position, processing_times, machines, num_jobs, num_machines)

            if current_makespan < particle.best_makespan:
                particle.best_makespan = current_makespan
                particle.best_position = particle.position.copy()

            if current_makespan < global_best_makespan:
                global_best_makespan = current_makespan
                global_best_position = particle.position.copy()

        loss_history.append(global_best_makespan)

        for particle in swarm:
            update_velocity(particle, global_best_position, inertia_weight, cognitive, social)
            update_position(particle)

    return global_best_makespan, global_best_position, loss_history

num_jobs = 4
num_machines = 4
processing_times = np.array([
    [54, 34, 61, 2],
    [9, 15, 89, 70],
    [38, 19, 28, 87],
    [95, 34, 7, 29]
])
machines = np.array([
    [3, 1, 4, 2],
    [4, 1, 2, 3],
    [1, 2, 3, 4],
    [1, 3, 2, 4]
])

best_makespan, best_schedule, loss_history = pso(processing_times, machines, num_jobs, num_machines)

print("best makespan:", best_makespan)
print("best schedule:", best_schedule)


plt.plot(loss_history)
plt.title('Makespan over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Best Makespan')
plt.grid(True)
plt.show()
