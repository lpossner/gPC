import pygpc
import h5py
from matplotlib import pyplot as plt
from collections import OrderedDict


folder = "data"
session_filename = "data/gPC.pkl"
results_filename = "data/gPC"

# read session
session = pygpc.read_session(fname=session_filename, folder=folder)

# read parameters
parameters = session.parameters_random

with h5py.File("data/gPC.hdf5", "r") as file:
    coeffs = file["coeffs"][:]
    results = file["model_evaluations/results"][:]

# Post-process gPC
pygpc.get_sensitivities_hdf5(fn_gpc=results_filename,
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             algorithm="standard")

# get a summary of the sensitivity coefficients
sobol, gsens = pygpc.get_sens_summary(results_filename, parameters)

print(sobol)
print(gsens)

# plot gPC approximation and results data
pygpc.plot_gpc(session=session,
               coeffs=coeffs,
               random_vars=session.problem.parameters_keys,
               n_grid=list(session.gpc[0].grid.coords.shape),
               coords=session.gpc[0].grid.coords,
               results=results,
               output_idx = [0]
               )

plt.savefig("gPC.png")
plt.show()
