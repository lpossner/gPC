import os
import pygpc
import numpy as np
from collections import OrderedDict


# seed RNG
seed = 1
np.random.seed(seed)

# define the properties of the random variables
parameters = OrderedDict()
# parameters["angle"] = pygpc.Beta(pdf_shape=[2, 2], pdf_limits=[-90, 90])
parameters["brightness"] = pygpc.Beta(pdf_shape=[2, 2], pdf_limits=[0, 2])

# create grid object
grid = pygpc.LHS(parameters_random=parameters, options={"seed": seed, "criterion": "ese"}, n_grid=1000)

# save grid.coords
os.makedirs("data", exist_ok=True)
np.save("data/coords.npy", grid.coords)
