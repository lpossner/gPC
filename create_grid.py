import os
import pygpc
import numpy as np
from collections import OrderedDict


# seed RNG
seed = 1
np.random.seed(seed)

# define the properties of the random variables
parameters = OrderedDict()
parameters["noise"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 10])

# create grid object
grid = pygpc.LHS(parameters_random=parameters, options={"seed": seed, "criterion": "ese"}, n_grid=100)

# save grid.coords
os.makedirs("data", exist_ok=True)
np.save("data/coords.npy", grid.coords)
