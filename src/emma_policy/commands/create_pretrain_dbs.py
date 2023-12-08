import argparse
from pathlib import Path

from emma_datasets.common import Settings

from emma_policy.datamodules.pretrain_instances import PreparePretrainInstancesDb


settings = Settings()


INSTANCES_DB_FILE_PATH = settings.paths.databases.joinpath("instances.db")


def create_pretrain_dbs(
    instances_db_file_path: Path,
    output_dir_path: Path,
    num_workers: int = 0,
    batch_size_per_worker: int = 5,
    max_mlm_valid_regions: int = 5,
) -> None:
    """Create all the pretrain instances DBs for every task in advance."""
    preparer = PreparePretrainInstancesDb(
        instances_db_file_path,
        output_dir_path,
        loader_num_workers=num_workers,
        loader_batch_size_per_worker=batch_size_per_worker,
        max_mlm_valid_regions=max_mlm_valid_regions,
    )

    preparer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create all the pretrain instances DBs for every task in advance."
    )

    parser.add_argument(
        "--instances-db-file-path",
        type=Path,
        default=INSTANCES_DB_FILE_PATH,
        help="Path to the `instances.db` file",
    )

    parser.add_argument(
        "--output-dir-path",
        type=Path,
        default=settings.paths.databases,
        help="Path where all DBs are saved",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers to process each batch faster. Larger is better.",
    )

    parser.add_argument(
        "--batch-size-per-worker",
        type=int,
        default=5,
        help="Batch size per worker. If too large, then you will likely run out of memory.",
    )

    parser.add_argument(
        "--max-mlm-valid-regions",
        type=int,
        default=5,
        help="Number of maximum region captions used for validation mlm.",
    )

    args = parser.parse_args()

    create_pretrain_dbs(
        args.instances_db_file_path,
        args.output_dir_path,
        args.num_workers,
        args.batch_size_per_worker,
        args.max_mlm_valid_regions,
    )
