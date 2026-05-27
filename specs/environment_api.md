# Environment Wrapper API

This file defines the public DSRC environment wrapper contract that all topologies should expose.

## Purpose

The public environment API should stay stable even if the underlying simulator changes. The wrapper owns:

- translating `highway_env` internals into DSRC project data structures
- returning AV-indexed mappings instead of tuple-based multi-agent outputs
- exposing topology, metric, and summary accessors that later baselines and RL code can rely on

## Public Interface

Every topology-specific environment wrapper should implement the following methods:

```python
reset(config=None, seed=None, options=None) -> tuple[dict[str, dict], dict]
step(av_actions) -> tuple[dict[str, dict], dict[str, float], bool, bool, dict]
get_local_observations() -> dict[str, dict]
get_global_state() -> dict
get_segment_metrics() -> dict[str, dict]
get_episode_summary() -> dict
```

## Method Contract

`reset(config=None, seed=None, options=None)`

- accepts an optional DSRC config override
- accepts an optional random seed
- returns `local_observations, info`
- `local_observations` is keyed by AV identifier, not by positional tuple index

`step(av_actions)`

- accepts one public action mapping keyed by AV identifier
- returns `local_observations, rewards, terminated, truncated, info`
- `rewards` is keyed by AV identifier
- `terminated` and `truncated` are episode-level booleans
- `av_actions`, `local_observations`, and `rewards` are keyed only by currently active AV identifiers
- inactive or exited AV identifiers should be rejected or ignored with an explicit diagnostic, depending on wrapper strictness
- blocked, masked, or modified actions should be reported through `info`

`get_local_observations()`

- returns the latest AV-indexed observation mapping without stepping the simulator
- excludes inactive or exited AVs

`get_global_state()`

- returns the centralized critic state for CTDE training
- may include topology-level and segment-level fields not visible to individual AVs
- active vehicle counts include only vehicles still on the topology

`get_segment_metrics()`

- returns one mapping per segment keyed by canonical segment identifier
- excludes inactive or exited vehicles from per-segment counts

`get_episode_summary()`

- returns a compact rollup used for CSV, JSON, and experiment summaries
- may include completed/exited vehicle counts

## Standard Identifiers

Topology IDs:

- `ring`
- `straight_single_lane`
- `straight_multilane`
- `merge`
- `inverted_tree`

Vehicle roles:

- `av`
- `human`

## Repo Ownership

The public interface should live in code under:

- `src/envs/base_ctde_env.py`
- `src/envs/wrappers.py`

Topology-specific implementations should later live under:

- `src/envs/`
- `src/road/`
