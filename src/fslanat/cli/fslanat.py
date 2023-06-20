from pathlib import Path

import click
import prefect_dask
from dask import config

from fslanat.flows.fslanat import fslanat_flow


def _main(
    anats: frozenset[Path] | None = None,
    output_dir: Path = Path("out"),
    n_workers: int = 1,
) -> None:
    # It would be prefereable to have paralellism that never spilled to the disk
    # Since we're using Dask, we'll just turn off that feature
    config.set(
        {"distributed.worker.memory.rebalance.measure": "managed_in_memory"}
    )
    config.set({"distributed.worker.memory.spill": False})
    config.set({"distributed.worker.memory.target": False})
    config.set({"distributed.worker.memory.pause": False})
    config.set({"distributed.worker.memory.terminate": False})
    config.set({"distributed.comm.timeouts.connect": "90s"})
    config.set({"distributed.comm.timeouts.tcp": "90s"})
    # config.set({"distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0})

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    fslanat_flow.with_options(
        task_runner=prefect_dask.DaskTaskRunner(
            cluster_kwargs={
                "n_workers": n_workers,
                "threads_per_worker": 1,
                "dashboard_address": None,
            }
        )
    )(images=anats, out=output_dir, return_state=True)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--bids-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option("--sub-limit", type=int, default=None)
@click.option("--n-workers", type=int, default=1)
def main(
    bids_dir: Path | None = None,
    output_dir: Path = Path("out"),
    sub_limit: int | None = None,
    n_workers: int = 1,
) -> None:
    anats = (
        frozenset(list(bids_dir.rglob("*T1w.nii.gz"))[:sub_limit])
        if bids_dir
        else None
    )
    _main(output_dir=output_dir, anats=anats, n_workers=n_workers)
