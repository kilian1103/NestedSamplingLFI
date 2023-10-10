from mpi4py import MPI

from NSLFI.NRE_Post_Analysis import plot_NRE_posterior, plot_NRE_expansion_and_contraction_rate
from NSLFI.utils import reload_data_for_plotting


def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    if rank_gen == 0:
        root_storage, network_storage, nreSettings, dataEnv = reload_data_for_plotting()
        plot_NRE_posterior(nreSettings=nreSettings, network_storage=network_storage, root_storage=root_storage,
                           dataEnv=dataEnv)
        if nreSettings.activate_NSNRE_counting:
            plot_NRE_expansion_and_contraction_rate(nreSettings=nreSettings, root_storage=root_storage)


if __name__ == '__main__':
    main()
