if __name__ == '__main__':

    import numpy as np
    import pygpc
    from collections import OrderedDict


    # seed RNG
    seed = 1
    np.random.seed(seed)

    # define parameters
    parameters = OrderedDict()
    parameters["noise"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 10])

    # load coords
    coords = np.load("data/coords.npy")
    # re-generate grid object from grid.coords
    grid = pygpc.LHS(parameters_random=parameters, coords=coords)
    # load results
    results = np.load("data/results.npy")

    # define gPC options
    options = OrderedDict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [10]
    options["order_max"] = 10
    options["interaction_order"] = 1
    options["error_type"] = "loocv"
    options["error_norm"] = "absolute"
    options["n_samples_validation"] = None
    options["fn_results"] = 'data/gPC'
    options["save_session_format"] = ".pkl"
    options["backend"] = "omp"
    options["verbose"] = True

    # determine number of gPC coefficients (hint: compare it with the amount of output data you have)
    # Tipp: 4x mehr sims als n_coeffs für stabile invertierung, Restfehler von Polynonordnung und Problem an sich
    n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                        order_glob_max=options["order_max"],
                                        order_inter_max=options["interaction_order"],
                                        dim=len(parameters))

    # define algorithm
    algorithm = pygpc.Static_IO(parameters=parameters, options=options, grid=grid, results=results)

    # initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()
