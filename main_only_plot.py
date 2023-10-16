from anesthetic import read_chains
from mpi4py import MPI

from NSLFI.NRE_Post_Analysis import plot_NRE_posterior, plot_quantile_plot
from NSLFI.utils import reload_data_for_plotting


def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    if rank_gen == 0:
        root_storage, network_storage, nreSettings, dataEnv = reload_data_for_plotting()
        plot_NRE_posterior(nreSettings=nreSettings, network_storage=network_storage, root_storage=root_storage,
                           dataEnv=dataEnv)

        for i in range(nreSettings.num_features):
            root = root_storage[f"round_{i}"]
            samples = read_chains(root=f"{root}/{nreSettings.file_root}")
            samples = samples.iloc[:, :nreSettings.num_features]
            plot_quantile_plot(samples=samples, percentiles=nreSettings.percentiles_of_quantile_plot,
                               nreSettings=nreSettings, root=root)


if __name__ == '__main__':
    main()
