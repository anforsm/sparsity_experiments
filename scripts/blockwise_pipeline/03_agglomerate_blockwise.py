import logging
import daisy
from funlib.geometry import Coordinate, Roi
from funlib.persistence import graphs, Array, open_ds, prepare_ds
import time
from utils import neighborhood
from lsd.post import agglomerate_in_block
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)

USE_PSQL = False
db_name = "anton_test5"


def agglomerate(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset: str,
    context: tuple,
    num_workers: int = 7,
    merge_function: str = "hist_quant_75",
) -> None:
    """Run agglomeration in parallel blocks. Requires that fragments and affinities have been predicted before.

    Args:
        affs_file (``str``):
            Path (relative or absolute) to the zarr file where affinities are stored.

        affs_dataset (``str``):
            The name of the fragments dataset to read from in the affinities file.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file where fragments are stored.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from in the fragments file.

        context (``tuple(int, int, int)``):
            The context to consider for fragment extraction and agglomeration, in world units.

        num_workers (``int``):
            How many blocks to run in parallel. Default is 7.

        merge_function (``str``):
            Symbolic name of a merge function. Default is hist_quant_75. See dictionary below.
    """

    start: float = time.time()
    logging.info(msg=f"Reading {affs_dataset} from {affs_file}")

    fragments: Array = open_ds(
        filename=fragments_file, ds_name=fragments_dataset, mode="r"
    )

    voxel_size: Coordinate = fragments.voxel_size
    total_roi: Roi = fragments.roi

    write_roi = daisy.Roi(offset=(0,) * 3, shape=Coordinate(fragments.chunk_shape))

    #context = Coordinate(context)

    write_roi: Roi = write_roi * voxel_size
    #read_roi: Roi = read_roi * voxel_size

    read_roi: Roi = (
        write_roi.grow(context, context)
    )

    #write_roi = Roi(offset=(0,) * 3, shape=total_roi.shape)
    #read_roi = Roi(offset=(0,) * 3, shape=total_roi.shape)

    logging.info(f"Reading fragments from {fragments_file}")


    total_roi = fragments.roi
    waterz_merge_function: dict = {
        "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
        "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
        "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
        "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
        "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
        "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
        "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
        "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
        "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
        "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
        "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    }[merge_function]

    logging.info(f"Reading affs from {affs_file}")
    affs: Array = open_ds(filename=affs_file, ds_name=affs_dataset)

    # opening RAG file
    logging.info(msg="Opening RAG file...")

    if not USE_PSQL:
        rag_provider = graphs.SQLiteGraphDataBase(
            Path("test.sqlite"),
            ["center_z", "center_y", "center_x"],
            mode="r+",
            edge_attrs={"merge_score": float, "agglomerated": bool},
            #nodes_collection="hglom_nodes",
            #edges_collection=f"hglom_edges_{merge_function}",
        )
    else:
        pass

    def worker(block):
        rag_provider = graphs.PgSQLGraphDatabase(
            db_name=db_name,
            db_host="khlabgpu.clm.utexas.edu",
            db_user="anton",
            db_password="password",
            db_port="5433",
            position_attributes=["center_z", "center_y", "center_x"],
            edge_attrs={"merge_score": float, "agglomerated": bool},
            mode="r+"
        )

        agglomerate_in_block(
            affs=affs,
            fragments=fragments,
            rag_provider=rag_provider,
            block=block,
            merge_function=waterz_merge_function,
            threshold=1.0,
        )

        


    logging.info(msg="RAG file opened")

    task = daisy.Task(
        task_id="AgglomerateTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=worker,
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError("at least one block failed!")

    end: float = time.time()

    seconds: float = end - start
    minutes: float = seconds / 60
    hours: float = minutes / 60
    days: float = hours / 24

    print(
        f"Total time to agglomerate fragments: {seconds} seconds / {minutes} minutes / {hours} hours / {days} days"
    )
    return done

if __name__ == "__main__":
    context = (100, 10, 10)

    agglomerate(
        affs_file="../../data/oblique.zarr",
        affs_dataset="test_200000_from_raw/s0",
        fragments_file="../../data/oblique.zarr",
        fragments_dataset="frags",
        context=context,
        num_workers=10,
    )