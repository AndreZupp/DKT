from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ParametricBuffer, ClassBalancedBuffer, HerdingSelectionStrategy


def herderSampling(mem_size):
    storage_policy = ParametricBuffer(max_size=mem_size, selection_strategy="HerdingSelectionStrategy")
    replay_plugin = ReplayPlugin(mem_size=mem_size, storage_policy=storage_policy)
    return replay_plugin


def random_sampling(mem_size):
    storage_policy = ParametricBuffer(max_size=mem_size, selection_strategy="RandomExemplarsSelectionStrategy")
    replay_plugin = ReplayPlugin(mem_size=mem_size, storage_policy=storage_policy)
    return replay_plugin


def class_balanced_sampling(mem_size):
    storage_policy = ClassBalancedBuffer(mem_size)
    replay_plugin = ReplayPlugin(mem_size, storage_policy=storage_policy)
    return replay_plugin


def class_herder_sampling(mem_size):
    storage_policy = ParametricBuffer(max_size=mem_size, groupby="class", selection_strategy=HerdingSelectionStrategy())
    replay_plugin = ReplayPlugin(mem_size, storage_policy=storage_policy)
    return replay_plugin
