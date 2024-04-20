import logging
import os

import sklearn
import swyft
import torch
from mpi4py import MPI
from swyft import collate_output as reformat_samples, Simulator
from torch import Tensor

from NSLFI.NRE_Settings import NRE_Settings


def retrain_next_round(root: str, training_data: Tensor, nreSettings: NRE_Settings,
                       sim: Simulator,
                       obs: swyft.Sample, network: swyft.SwyftModule, trainer: swyft.SwyftTrainer,
                       rd: int) -> swyft.SwyftModule:
    logger = logging.getLogger(nreSettings.logger_name)
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    logger.info(f"Simulating new {nreSettings.obsKey} using NS samples {nreSettings.targetKey} with Simulator!")

    ### simulate joint distribution using deadpoints ###
    samples = []
    if rank_gen == 0:
        for point in training_data:
            cond = {nreSettings.targetKey: point.float()}

            ### noise resampling ###
            if nreSettings.use_noise_resampling and rd > 0:
                resampler = sim.get_resampler(targets=[nreSettings.obsKey])
                for _ in range(nreSettings.n_noise_resampling_samples):
                    cond[nreSettings.obsKey] = None
                    sample = resampler(cond)
                    samples.append(sample)
            else:
                sample = sim.sample(conditions=cond, targets=[nreSettings.obsKey])
                samples.append(sample)
        logger.info(f"Total number of samples for training: {len(samples)}")
        samples = sklearn.utils.shuffle(samples)
    else:
        logger.info(f"Core {rank_gen}: Simulating training data set! Waiting...")

    comm_gen.Barrier()
    samples = comm_gen.bcast(samples, root=0)
    samples = reformat_samples(samples)

    ### save training data for NRE on disk ###
    if nreSettings.save_joint_training_data and rank_gen == 0:
        if nreSettings.use_livepoint_increasing:
            try:
                os.makedirs(f"{root}/{nreSettings.increased_livepoints_fileroot}")
            except OSError:
                logger.info("root folder already exists!")
            torch.save(
                f=f"{root}/{nreSettings.increased_livepoints_fileroot}/{nreSettings.joint_training_data_fileroot}",
                obj=samples)
        else:
            torch.save(f=f"{root}/{nreSettings.joint_training_data_fileroot}", obj=samples)

    comm_gen.Barrier()

    ### train network ###
    training_data_swyft = swyft.Samples(samples)
    logger.info("Simulation done!")
    logger.info("Setting up network for training!")
    # network = torch.compile(network)
    logger.info("Starting training!")
    dm = swyft.SwyftDataModule(data=training_data_swyft, **nreSettings.dm_kwargs)
    trainer.fit(network, dm)
    logger.info("Training done!")

    return network
