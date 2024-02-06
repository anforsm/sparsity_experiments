import logging
import multiprocessing as mp
import numpy as np
from funlib.geometry import Roi
from funlib.persistence import open_ds, graphs, Array
import os
import time
from funlib.segment.graphs.impl import connected_components
from pathlib import Path

USE_PSQL = False
db_name = "anton_test5"

logging.getLogger().setLevel(logging.INFO)


def find_segments(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset: str,
    thresholds_minmax: list = [0, 1],
    thresholds_step: float = 0.02,
    merge_function: str = "hist_quant_75",
) -> bool:
    """
    Extract and store fragments from supervoxels and generate segmentation lookup tables.

    Args:
        affs_file (``str``):
            Path (relative or absolute) to the zarr file where affinities are stored.

        affs_dataset (``str``):
            The name of the fragments dataset to read from in the affinities file.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file where fragments are stored.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from in the fragments file.

        thresholds_minmax (``list[int]``, optional):
            The lower and upper bounds to use for generating thresholds. Default is [0, 1].

        thresholds_step (``float``, optional):
            The step size to use when generating thresholds between min/max. Default is 0.02.

        merge_function (``str``, optional):
            The merge function used to create the segmentation. Default is "hist_quant_75".

    Returns:
        ``bool``:
            True if the operation was successful, False otherwise.
    """

    logging.info("Reading graph")
    start: float = time.time()

    fragments: Array = open_ds(fragments_file, fragments_dataset)

    affs: Array = open_ds(filename=affs_file, ds_name=affs_dataset)
    roi: Roi = affs.roi

    if not USE_PSQL:
        graph_provider = graphs.SQLiteGraphDataBase(
            Path("test.sqlite"),
            ["center_z", "center_y", "center_x"],
            mode="r",
            edge_attrs={"merge_score": float, "agglomerated": bool},
            #nodes_collection="hglom_nodes",
            #edges_collection=f"hglom_edges_{merge_function}",
        )
    else:
        graph_provider = graphs.PgSQLGraphDatabase(
            db_name=db_name,
            db_host="khlabgpu.clm.utexas.edu",
            db_user="anton",
            db_password="password",
            db_port="5433",
            position_attributes=["center_z", "center_y", "center_x"],
            edge_attrs={"merge_score": float, "agglomerated": bool},
            mode="r"
        )

    node_attrs: list = graph_provider.read_nodes(roi=roi)
    edge_attrs: list = graph_provider.read_edges(roi=roi, nodes=node_attrs)

    logging.info(msg=f"Read graph in {time.time() - start}")

    if "id" not in node_attrs[0]:
        logging.info(msg="No nodes found in roi %s" % roi)
        return

    nodes: list = [node["id"] for node in node_attrs]

    edge_u: list = [np.uint64(edge["u"]) for edge in edge_attrs]
    edge_v: list = [np.uint64(edge["v"]) for edge in edge_attrs]

    edges: np.ndarray = np.stack(arrays=[edge_u, edge_v], axis=1)

    scores: list = [np.float32(edge["merge_score"]) for edge in edge_attrs]

    logging.info(msg=f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

    out_dir: str = os.path.join(fragments_file, "luts", "fragment_segment")

    os.makedirs(out_dir, exist_ok=True)

    thresholds = [
        round(i, 3)
        for i in np.arange(
            float(thresholds_minmax[0]), float(thresholds_minmax[1]), thresholds_step
        )
    ]

    # parallel processing
    start = time.time()

    with mp.Pool(processes=1) as pool:
        pool.starmap(
            get_connected_components,
            [
                (
                    np.asarray(nodes, dtype=np.uint64),
                    np.asarray(edges, dtype=np.uint64),
                    np.asarray(scores),
                    t,
                    f"hglom_edges_{merge_function}",
                    out_dir,
                )
                for t in thresholds
            ],
        )

    logging.info(f"Created and stored lookup tables in {time.time() - start}")

    return True


def get_connected_components(
    nodes,
    edges,
    scores,
    threshold,
    edges_collection,
    out_dir,
):
    logging.info(f"Getting CCs for threshold {threshold}...")
    components = connected_components(nodes, edges, scores, threshold)

    logging.info(f"Creating fragment-segment LUT for threshold {threshold}...")
    lut = np.array([nodes, components])

    logging.info(f"Storing fragment-segment LUT for threshold {threshold}...")

    lookup = f"seg_{edges_collection}_{int(threshold*1000)}"

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)

if __name__ == "__main__":
    find_segments(
        affs_file="../../data/oblique.zarr",
        affs_dataset="test_200000_from_raw/s0",
        fragments_file="../../data/oblique.zarr",
        fragments_dataset="frags",
        thresholds_minmax=[0, 1],
        thresholds_step=0.02
    )
