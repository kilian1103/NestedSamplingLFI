import numpy as np
import sklearn
import swyft
import torch
from swyft import collate_output as reformat_samples, Simulator
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


def retrain_next_round(root: str, deadpoints: np.ndarray, polyswyftSettings: PolySwyft_Settings,
                       sim: Simulator, network: swyft.SwyftModule, trainer: swyft.SwyftTrainer,
                       rd: int) -> swyft.SwyftModule:
    """
    Retrain the network for the next round of NSNRE.
    :param root: A string of the root folder
    :param deadpoints: A tensor of deadpoints
    :param polyswyftSettings: A PolySwyft_Settings object
    :param sim: A swyft simulator object
    :param obs: A swyft sample of the observed data
    :param network: A swyft network object
    :param trainer: A swyft trainer object
    :param rd: An integer of the round number
    :return: A trained swyft network object
    """
    logger = polyswyftSettings.logger
    try:
        from mpi4py import MPI
    except ImportError:
        raise ImportError("mpi4py is required for this function!")

    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    logger.info(
        f"Simulating joint training dataset ({polyswyftSettings.obsKey}, {polyswyftSettings.targetKey}) using deadpoints with "
        f"Simulator!")

    ### simulate joint distribution using deadpoints ###
    deadpoints = np.array_split(deadpoints, size_gen)
    deadpoints = deadpoints[rank_gen]
    samples = []
    for point in deadpoints:
        cond = {polyswyftSettings.targetKey: point}
        ### noise resampling ###
        if polyswyftSettings.use_noise_resampling and rd > 0:
            resampler = sim.get_resampler(targets=[polyswyftSettings.obsKey])
            for _ in range(polyswyftSettings.n_noise_resampling_samples):
                cond[polyswyftSettings.obsKey] = None
                sample = resampler(cond)
                samples.append(sample)
        else:
            sample = sim.sample(conditions=cond, targets=[polyswyftSettings.obsKey])
            samples.append(sample)

    del deadpoints
    comm_gen.Barrier()
    samples = comm_gen.allgather(samples)
    samples = np.concatenate(samples, axis=0)
    samples = samples.tolist()
    logger.info(f"Total number of samples for training the network: {len(samples)}")
    comm_gen.Barrier()
    samples = reformat_samples(samples)
    logger.info("Simulation done!")

    ### concatenate previous training data with new training data###
    if rd > 0:
        previous_root = f"{polyswyftSettings.root}/{polyswyftSettings.child_root}_{rd - 1}"
        try:
            previous_samples = torch.load(f=f"{previous_root}/{polyswyftSettings.joint_training_data_fileroot}")
        except:
            raise FileNotFoundError(
                f"Could not find joint training data for round {rd - 1}! Activate "
                f"polyswyftSettings.save_joint_training_data = True and restart!")

        thetas = np.concatenate((previous_samples[polyswyftSettings.targetKey], samples[polyswyftSettings.targetKey]), axis=0)
        Ds = np.concatenate((previous_samples[polyswyftSettings.obsKey], samples[polyswyftSettings.obsKey]), axis=0)
        del previous_samples
        samples = {polyswyftSettings.targetKey: thetas, polyswyftSettings.obsKey: Ds}

    ### shuffle training data to reduce ordinal bias introduced by deadpoints###
    if rank_gen == 0:
        thetas = samples[polyswyftSettings.targetKey]
        Ds = samples[polyswyftSettings.obsKey]
        joint =  np.array(sklearn.utils.shuffle(list(zip(thetas,Ds))),dtype=object)
        thetas = np.stack(joint[:, 0])
        Ds = np.stack(joint[:, 1])
        del joint
        samples = {polyswyftSettings.targetKey: thetas, polyswyftSettings.obsKey: Ds}
        del thetas, Ds

    comm_gen.Barrier()
    samples = comm_gen.bcast(samples, root=0)

    ### save training data for NRE on disk ###
    if polyswyftSettings.save_joint_training_data and rank_gen == 0:
        torch.save(f=f"{root}/{polyswyftSettings.joint_training_data_fileroot}", obj=samples)

    comm_gen.Barrier()

    ### train network ###
    samples = swyft.Samples(samples)
    # network = torch.compile(network)
    logger.info("Starting training of network!")
    dm = swyft.SwyftDataModule(data=samples, **polyswyftSettings.dm_kwargs)
    del samples
    network.train()
    trainer.fit(network, dm)
    logger.info("Training done!")

    return network
