from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
sns.set()

# Some constants
FRAME_RATE = 10          # Refresh graphics very FRAME_RATE hours
DENSITY = 200            #Number of people on the map
I0 = 0.03                #probability of being infected at t=0
SOCIAL_DISTANCE = 0.007 # in km
SPEED = 6               # km/day
SPEED_W = SPEED # Half of a normal person
AREA_SIZE = 1000 # in m
SIGMA = (SPEED * 1000. / 24. / AREA_SIZE) / (3. * np.sqrt(2)) # in meter per hour
SIGMA_W = (SPEED_W * 1000. / 24. / AREA_SIZE) / (3. * np.sqrt(2)) # in meter per hour
BETA1 = 0.5             # Probality to gets infected (From "S" to "I")
BETA2 = 0.75            # Probability to get infecte when in a cluster (From "S" to "I")
BETA3 = 0.8             # Probality to gets infected when near a walking dead (From "S" to "I")
GAMMA1 = 7 * 24         # Number of hours before recovering (From "I" to "R")
GAMMA2 = 0.003          # Probability to die (From "I" to "D")
GAMMA3 = 0.003          # probablity to become a walking dead (from "I" to "W")
GAMMA_W = 4 * 24        # Number of hours before a WalkingDead dies
EPSILON = 0.05          # Probability to be Susceptible again (From "R" to "S")
BORDER = False
LOCKDOWN = False
MAX_HOME_DISTANCE = 0.1
CLUSTURING = True
WALKING_DEAD = False

# For graph
UNSEEN = 0
DONE = 1

# Get a and b of the line equation (ax + b) from 2 points
def compute_line_parameters(p1, p2):
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - a * p1[0]
    return a, b

## The locations of borders on the map
A = (0., 0.82142857)
B = (0.46896552,0.53125)
C = (0.57241379,0.58928571)
D = (1.,0.33035714)
LINE1 = compute_line_parameters(A, B)
LINE2 = compute_line_parameters(B, C)
LINE3 = compute_line_parameters(C, D)


class SIRState(Enum):
    SUSCEPTIBLE = 0
    INFECTIOUS = 1
    RECOVERED = 2
    DEAD = 3
    WALKING_DEAD = 4

class District(Enum):
    D7 = 0
    D15 = 1

def distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

@dataclass
class Person:
    x: float        # Normalized x position
    y: float        # normalized y position
    original_x: float # "home x"
    original_y: float # "home y"
    district: District
    succ = []         # list of neigbors
    status = UNSEEN   # for cluster nodes computation
    is_in_infectious_cluster: bool 

    def __init__(self, x, y):
        self.x = x
        self.original_x = x
        self.y = y
        self.original_y = y
        self.district = compute_district(x, y)
        self.is_in_infectious_cluster = False

    # Function that tells if the person has infected_neightbors
    def has_infected_neighbor(self):
        for neighbor in self.succ:
            if BORDER and neighbor.district != self.district:
                continue
            elif (neighbor.state == SIRState.INFECTIOUS):
                return SIRState.INFECTIOUS
            elif (neighbor.state == SIRState.WALKING_DEAD):
                return SIRState.WALKING_DEAD

        return None

    def move(self, sigma=SIGMA):
        dx = np.random.normal(0, SIGMA)
        dy = np.random.normal(0, SIGMA)

        # Clip to borders
        new_x = np.clip(self.x + dx, 0.0, 1.0) #if out of the map --> cliped to the edge
        new_y = np.clip(self.y + dy, 0.0, 1.0)

        if (LOCKDOWN and distance(new_x, new_y, self.original_x, self.original_y) >= MAX_HOME_DISTANCE) or \
            (BORDER and compute_district(new_x, new_y) != self.district):
            return

        self.x = new_x
        self.y = new_y

    def update(self):
        return self

# A Susceptible that moves randomly
class SusceptiblePerson(Person):
    state = SIRState.SUSCEPTIBLE

    def update(self):
        infectious = self.has_infected_neighbor() #infectious neigbor ?

        if (infectious == SIRState.INFECTIOUS and np.random.rand() < BETA1) or \
                (infectious == SIRState.WALKING_DEAD and np.random.rand() < BETA3) or \
                (CLUSTURING is True and self.is_in_infectious_cluster is True and np.random.rand() < BETA2):
            return InfectiousPerson(self.x, self.y)
        return self


class InfectiousPerson(Person):
    age = 0 #reset age for recovery time
    state = SIRState.INFECTIOUS

    def update(self):
        self.age += 1
        if np.random.rand() < GAMMA2:
            return DeadPerson(self.x, self.y)
        elif WALKING_DEAD is True and np.random.rand() < GAMMA3:
            return WalkingDeadPerson(self.x, self.y)
        elif self.age > GAMMA1:
            return RecoveredPerson(self.x, self.y)
        return self

class WalkingDeadPerson(Person):
    age = 0 #reset age for time left before dying
    state = SIRState.WALKING_DEAD

    def move(self):
        super().move(sigma=SIGMA_W) #override move function with sigma w (walkers are slower)

    def update(self):
        self.age += 1
        if self.age > GAMMA_W:
            return DeadPerson(self.x, self.y)
        return self

class DeadPerson(Person):
    state = SIRState.DEAD

    def move(self):
        pass #dead person does not move

    def update(self):
        return self

class RecoveredPerson(Person):
    state = SIRState.RECOVERED

    def update(self):
        if np.random.rand() < EPSILON:
            return SusceptiblePerson(self.x, self.y)
        return self

#return matrix of the distances between people
def to_matrix(people): 
    size = len(people) #size of the matrix  : N people x N people
    out = np.zeros((size, size)) #matrix filled with zeroes
    for i in range(size):
        for j in range(i + 1, size): #no need to compute distance with itself
            dist = distance(people[i].x, people[i].y, people[j].x, people[j].y)
            out[i][j] = dist  #fill half the matrix with distances
            out[j][i] = dist # fill other half

    return out
#all cluster even if not infectious
def get_cluster(p):
    out = [p] #list of p 
    infectious_count = 1 if (p.state == SIRState.INFECTIOUS or p.state == SIRState.WALKING_DEAD) else 0 #count infectious people in cluster
    p.status = DONE # does not go through the same nod again

    for neighbor in p.succ: #each neigbor 
        if neighbor.status == DONE or (BORDER and neighbor.district != p.district): #reasons not to compute this node
            continue
        neighbor_infectious, neighbor_cluster = get_cluster(neighbor) #next neigbor's environment
        infectious_count += neighbor_infectious #update count
        out.extend(neighbor_cluster)

    return infectious_count, out

#filter clusters 
def get_infectious_clusters(people):
    out = []

    for p in people:
        if p.status == DONE:
            continue
        infectious_count, cluster = get_cluster(p)
        infectious_percentage = float(infectious_count) / len(cluster) #percentage of infected in cluster

        if len(cluster) < 5 or infectious_percentage < 0.33: #filter
            continue
        out.append(cluster)

    return out

def update_graph(people):
    # Reset everything
    for p in people:
        p.succ = []
        p.status = UNSEEN
        p.is_in_infectious_cluster = False

    adjacency_matrix = to_matrix(people)
    size = len(people)
    for i in range(size):
        for j in range(i + 1, size):
            if adjacency_matrix[i][j] >= SOCIAL_DISTANCE or people[j].status == SIRState.DEAD:
                continue # Skip
            people[i].succ.append(people[j]) #fill the succ list with neigbors
            people[j].succ.append(people[i])

    if CLUSTURING is True:
            infectious_clusters = get_infectious_clusters(people)
            for cluster in infectious_clusters:
                for p in cluster:
                    p.is_in_infectious_cluster = True

def compute_district(x, y):
    if x < B[0]: # to the left of B
        if y > x * LINE1[0] + LINE1[1]: #above AB
            return District.D7
        return District.D15
    elif x > C[0]: # to the right of C
        if y > x * LINE3[0] + LINE3[1]: #above CD
            return District.D7
        return District.D15
    else: # between B and C
        if y > x * LINE2[0] + LINE2[1]: # above BC
            return District.D7
        return District.D15


'''
Fonctions used to display and plot the curves
(you should not have to change them)
'''

def display_map(people, ax = None):
    x = [ p.x for p in people]
    y = [ p.y for p in people]
    h = [ p.state.name[0] for p in people]
    horder = ["S", "I", "R", "D", "W"]
    ax = sns.scatterplot(x, y, hue=h, hue_order=horder, ax=ax)
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((0.0,1.0))
    ax.set_aspect(224/145)
    ax.set_axis_off()
    ax.set_frame_on(True)
    ax.legend(loc=1, bbox_to_anchor=(0, 1))


count_by_population = None
def plot_population(people, ax = None):
    global count_by_population

    states = np.array([p.state.value for p in people], dtype=int)
    counts = np.bincount(states, minlength=5)
    entry = {
        "Susceptible" : counts[SIRState.SUSCEPTIBLE.value],
        "Infectious" : counts[SIRState.INFECTIOUS.value],
        "Dead" : counts[SIRState.DEAD.value],
        "Recovered" : counts[SIRState.RECOVERED.value],
        "Walking Dead": counts[SIRState.WALKING_DEAD.value]
    }
    cols = ["Susceptible", "Infectious", "Recovered", "Dead", "Walking Dead"]
    if count_by_population is None:
        count_by_population = pd.DataFrame(entry, index=[0.])
    else:
        count_by_population = count_by_population.append(entry, ignore_index=True)
    if ax != None:
        count_by_population.index = np.arange(len(count_by_population)) / 24
        sns.lineplot(data=count_by_population, ax = ax)


'''
Main loop function, that is called at each turn
'''
def next_loop_event(t):
    print("Time =",t)

    # Move each person
    for p in people:
        p.move()

    update_graph(people)

    # Update the state of people
    for i in range(len(people)):
        people[i] = people[i].update()

    if t % FRAME_RATE == 0:
        fig.clf()
        ax1, ax2 = fig.subplots(1,2)
        display_map(people, ax1)
        plot_population(people, ax2)
    else:
        plot_population(people, None)


'''
Function that crate the initial population
'''
def create_data():
    data = []
    for _ in range(DENSITY):
        x = np.random.rand()
        y = np.random.rand()
        if np.random.rand() < I0:
            to_add = InfectiousPerson(x, y)
        else:
            to_add = SusceptiblePerson(x, y)
        data.append(to_add)

    return data


import matplotlib.animation as animation

people = create_data()

fig = plt.figure(1)
duration = 20 # in days
anim = animation.FuncAnimation(fig, next_loop_event, frames=np.arange(duration*24), interval=100, repeat=False)

# To save the animation as a video
anim.animation.save("simulation.mp4", animation.FFMpegWriter(fps=5))

plt.show()