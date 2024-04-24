import os

import sklearn
import swyft
import torch
from swyft import collate_output as reformat_samples, Simulator
from torch import Tensor

from NSLFI.NRE_Settings import NRE_Settings


def retrain_next_round(root: str, deadpoints: Tensor, nreSettings: NRE_Settings,
                       sim: Simulator, network: swyft.SwyftModule, trainer: swyft.SwyftTrainer,
                       rd: int) -> swyft.SwyftModule:
    """
    Retrain the network for the next round of NSNRE.
    :param root: A string of the root folder
    :param deadpoints: A tensor of deadpoints
    :param nreSettings: A NRE_Settings object
    :param sim: A swyft simulator object
    :param obs: A swyft sample of the observed data
    :param network: A swyft network object
    :param trainer: A swyft trainer object
    :param rd: An integer of the round number
    :return: A trained swyft network object
    """
    logger = nreSettings.logger
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("mpi4py is required for this function!")

    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    logger.info(
        f"Simulating joint training dataset ({nreSettings.obsKey}, {nreSettings.targetKey}) using deadpoints with "
        f"Simulator!")

    ### simulate joint distribution using deadpoints ###
    samples = []
    if rank_gen == 0:
        for point in deadpoints:
            cond = {nreSettings.targetKey: point.float()}

            ### noise resampling ###
            if nreSettings.use_noise_resampling:
                resampler = sim.get_resampler(targets=[nreSettings.obsKey])
                for _ in range(nreSettings.n_noise_resampling_samples):
                    cond[nreSettings.obsKey] = None
                    sample = resampler(cond)
                    samples.append(sample)
            else:
                sample = sim.sample(conditions=cond, targets=[nreSettings.obsKey])
                samples.append(sample)
        logger.info(f"Total number of samples for training the network: {len(samples)}")
        samples = sklearn.utils.shuffle(samples)
    else:
        logger.info(f"Core {rank_gen}: Simulating training data set! Waiting...")

    comm_gen.Barrier()
    samples = comm_gen.bcast(samples, root=0)
    samples = reformat_samples(samples)
    logger.info("Simulation done!")

    ### save training data for NRE on disk ###
    if nreSettings.save_joint_training_data and rank_gen == 0:
        torch.save(f=f"{root}/{nreSettings.joint_training_data_fileroot}", obj=samples)

    comm_gen.Barrier()

    ### train network ###
    training_data_swyft = swyft.Samples(samples)
    # network = torch.compile(network)
    logger.info("Starting training of network!")
    dm = swyft.SwyftDataModule(data=training_data_swyft, **nreSettings.dm_kwargs)
    network.train()
    trainer.fit(network, dm)
    logger.info("Training done!")

    return network
