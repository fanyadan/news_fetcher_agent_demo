from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _int_env(*keys: str) -> Optional[int]:
    for k in keys:
        v = os.getenv(k)
        if v is None:
            continue
        v = v.strip()
        if not v:
            continue
        try:
            return int(v)
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class DistributedContext:
    """Best-effort process context for distributed/multi-process execution."""

    rank: int
    world_size: int
    local_rank: Optional[int]

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 or self.rank > 0 or (self.local_rank or 0) > 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def get_distributed_context() -> DistributedContext:
    """Infer (rank, world_size, local_rank) from common launcher environment variables.

    Supports torchrun, Slurm, OpenMPI/MPI, and manual overrides via:
    - NEWS_AGENT_RANK
    - NEWS_AGENT_WORLD_SIZE
    - NEWS_AGENT_LOCAL_RANK
    """

    # Manual overrides (take precedence)
    rank = _int_env("NEWS_AGENT_RANK")
    world_size = _int_env("NEWS_AGENT_WORLD_SIZE")
    local_rank = _int_env("NEWS_AGENT_LOCAL_RANK")

    # torchrun / pytorch elastic
    if rank is None:
        rank = _int_env("RANK")
    if world_size is None:
        world_size = _int_env("WORLD_SIZE")
    if local_rank is None:
        local_rank = _int_env("LOCAL_RANK")

    # Slurm
    if rank is None:
        rank = _int_env("SLURM_PROCID")
    if world_size is None:
        world_size = _int_env("SLURM_NTASKS")
    if local_rank is None:
        local_rank = _int_env("SLURM_LOCALID")

    # MPI / OpenMPI / MVAPICH
    if rank is None:
        rank = _int_env("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK")
    if world_size is None:
        world_size = _int_env("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")
    if local_rank is None:
        local_rank = _int_env("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID")

    # Best-effort defaults.
    if rank is None:
        # If only local_rank is set (common), at least avoid duplicating work within a node.
        rank = int(local_rank or 0)

    if world_size is None:
        # If any rank/local_rank indicates multiple processes, assume >1.
        if (rank or 0) > 0 or (local_rank or 0) > 0:
            world_size = max(int(rank or 0), int(local_rank or 0)) + 1
        else:
            world_size = 1

    return DistributedContext(rank=int(rank), world_size=int(world_size), local_rank=local_rank)


def should_run_on_this_rank(*, default: str = "rank0") -> bool:
    """Return whether this process should perform side-effecting work.

    default:
      - "rank0": only rank 0 runs when distributed
      - "all": all ranks run

    Override via NEWS_AGENT_DISTRIBUTED_MODE.
    """

    mode = (os.getenv("NEWS_AGENT_DISTRIBUTED_MODE") or default).strip().lower()
    ctx = get_distributed_context()

    if not ctx.is_distributed:
        return True

    if mode in {"all", "every", "everyone"}:
        return True

    # rank0-only by default
    return ctx.is_main


def is_truthy_env(var_name: str, *, default: bool = False) -> bool:
    v = os.getenv(var_name)
    if v is None:
        return default
    return v.strip().lower() in _TRUE_VALUES


def rank0_print(*args: object, **kwargs: object) -> None:
    """Print only from rank 0 in distributed settings (unless mode=all)."""

    if should_run_on_this_rank():
        print(*args, **kwargs)


def mpi_sanity_check(comm: object, *, verbose: bool = False, tag: str = "") -> Optional[str]:
    """Lightweight best-effort MPI communication check.

    This is intended for debugging when launching under MPI (via mpi4py).

    It performs:
    - A ring send/recv (point-to-point)
    - An allgather of basic rank metadata
    - A barrier

    Returns an error string if the check fails, otherwise None.
    """

    try:
        if comm is None:
            return "MPI check failed: comm is None"

        rank = int(comm.Get_rank())
        world_size = int(comm.Get_size())

        if world_size <= 1:
            return None

        # Point-to-point ring check.
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size
        sent = {"from": rank, "to": next_rank}
        received = comm.sendrecv(sent, dest=next_rank, source=prev_rank)
        if not isinstance(received, dict) or int(received.get("from", -1)) != prev_rank:
            return (
                f"MPI ring check failed on rank {rank}: expected message from rank {prev_rank}, "
                f"got {received!r}"
            )

        # Collective allgather check.
        info = {"rank": rank, "pid": os.getpid(), "host": socket.gethostname()}
        gathered = comm.allgather(info)

        if not isinstance(gathered, list) or len(gathered) != world_size:
            got_len = len(gathered) if isinstance(gathered, list) else "n/a"
            return (
                f"MPI allgather check failed on rank {rank}: expected list of len {world_size}, "
                f"got {type(gathered)} len {got_len}"
            )

        ranks: List[int] = []
        hosts: Dict[str, int] = {}
        for item in gathered:
            if not isinstance(item, dict):
                continue
            r = item.get("rank")
            if isinstance(r, int):
                ranks.append(r)
            h = item.get("host")
            if isinstance(h, str):
                hosts[h] = hosts.get(h, 0) + 1

        if sorted(ranks) != list(range(world_size)):
            return (
                f"MPI allgather ranks mismatch on rank {rank}: expected ranks 0..{world_size - 1}, "
                f"got {sorted(ranks)}"
            )

        # Barrier so we know all ranks reached the check.
        comm.Barrier()

        if verbose and rank == 0:
            label = f"[mpi-check:{tag}]" if tag else "[mpi-check]"
            print(f"{label} ok: world_size={world_size}, hosts={hosts}")

        return None
    except Exception as e:
        return f"MPI check failed: {e}"


def get_rank_and_world_size() -> Tuple[int, int]:
    ctx = get_distributed_context()
    return ctx.rank, ctx.world_size
