"""Self-evolving visual reasoning agent utilities."""

from open_r1.self_evolve.failure_tags import assign_failure_tag, assign_failure_tags
from open_r1.self_evolve.iteration import score_and_route_trajectories
from open_r1.self_evolve.online_solver import OnlineSolveConfig, OnlineTransformersSolverSampler
from open_r1.self_evolve.rewards import compute_group_reward_vectors, compute_reward_vector

__all__ = [
    "assign_failure_tag",
    "assign_failure_tags",
    "compute_group_reward_vectors",
    "compute_reward_vector",
    "OnlineSolveConfig",
    "OnlineTransformersSolverSampler",
    "score_and_route_trajectories",
]
