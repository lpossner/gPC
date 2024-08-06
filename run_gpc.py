if __name__ == '__main__':

    import numpy as np
    import pygpc
    from collections import OrderedDict


    # seed RNGs
    seed = 1
    np.random.seed(seed)

    # define parameters
    parameters = OrderedDict()
    parameters["angle"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-45, 45])
    parameters["brightness"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.5, 2])

    # load coords
    coords = np.load("data/coords.npy")
    # re-generate grid object from grid.coords
    grid = pygpc.Random(parameters_random=parameters, coords=coords)
    # load results
    results = np.load("data/results.npy")

    # load validation grid
    coords_validation = np.load("data/coords_validation.npy")
    # re-generate validation grid object from grid.coords
    grid_validation = pygpc.Random(parameters_random=parameters, coords=coords_validation)
    # load validation results
    results_validation = np.load("data/results_validation.npy")
    # create validation set
    validation = pygpc.ValidationSet(grid=grid_validation, results=results_validation)

    # define gPC options
    options = OrderedDict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [15, 15]
    options["order_max"] = 15
    options["interaction_order"] = 2
    options["error_type"] = "nrmsd"
    options["error_norm"] = "relative"
    options["n_samples_validation"] = 1000
    options["fn_results"] = 'data/gPC'
    options["save_session_format"] = ".pkl"
    options["backend"] = "omp"
    options["verbose"] = True

    # determine number of gPC coefficients (hint: compare it with the amount of output data you have)
    # n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                     order_glob_max=options["order_max"],
    #                                     order_inter_max=options["interaction_order"],
    #                                     dim=len(parameters))

    # define algorithm
    algorithm = pygpc.Static_IO(parameters=parameters, options=options, grid=grid, results=results, validation=validation)

    # initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()
