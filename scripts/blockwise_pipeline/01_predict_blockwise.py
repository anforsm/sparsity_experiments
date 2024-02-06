import multiprocessing
multiprocessing.set_start_method('fork')

from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds
from pathlib import Path
import daisy
import json
import logging
import os
import subprocess
import time
import gunpowder as gp

logging.getLogger().setLevel(logging.INFO)


def predict_blockwise(config):
    config_path = config["config_path"]
    worker_path = config["worker_path"]
    iteration = config["iteration"]
    raw_file = config["raw_file"]
    raw_dataset = config["raw_dataset"]
    out_file = config["out_file"]
    num_workers = config["num_workers"]


    raw = gp.ArrayKey('RAW')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    # from here on, all values are in world units (unless explicitly mentioned)
    # get ROI of source
    source = gp.ZarrSource(raw_file,{raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)})

    # load config
    with open(os.path.join(config_path)) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(config_path)
        )
        net_config = json.load(f)
    outputs = net_config["outputs"]

    voxel_size = Coordinate([50,2,2])
    # get chunk size and context
    net_input_size = Coordinate(net_config["input_shape"]) * voxel_size 
    net_output_size = Coordinate(net_config["output_shape"]) * voxel_size 
    context = (net_input_size - net_output_size) / 2

    # get total input and output ROIs
    #input_roi = source.roi.grow(context, context)
    #output_roi = source.roi

    with gp.build(source):
        input_roi = source.spec[raw].roi
        output_roi = source.spec[raw].roi.grow(-context,-context)

    # create read and write ROI
    ndims = len(input_roi.shape)
    block_read_roi = Roi((0,) * ndims, net_input_size) - context
    block_write_roi = Roi((0,) * ndims, net_output_size)

    logging.info("Preparing output dataset...")

    for output_name, val in outputs.items():
        out_dims = val["out_dims"]
        out_dtype = val["out_dtype"]

        out_dataset = f"{output_name}_{iteration}_from_{raw_dataset}"
        print(f"Created output {output_name}: {out_file}/{out_dataset}")

        prepare_ds(
            out_file,
            out_dataset,
            output_roi,
            voxel_size,
            out_dtype,
            write_size=block_write_roi.shape,
            compressor={"id": "blosc"},
            delete=True,
            num_channels=out_dims,
        )

    logging.info("Starting block-wise processing...")

    predict_worker = os.path.abspath(os.path.join(worker_path, "01_predict.py"))

    logging.info(f"predict_worker: {predict_worker}")

    # process block-wise
    task = daisy.Task(
        "PredictBlockwiseTask",
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: start_worker(config, predict_worker),
        check_function=None,
        num_workers=num_workers,
        read_write_conflict=False,
        max_retries=5,
        fit="overhang",
    )

    done = daisy.run_blockwise([task])

    if not done:
        raise RuntimeError("at least one block failed!")


def start_worker(config: dict, worker: str):
    config_file = Path(config["out_file"]) / "config.json"

    with open(config_file, "w") as f:
        json.dump(config, f)
    
    logging.info("Running block with config %s..." % config_file)

    subprocess.run(
        [
            "python",
            worker,
            str(config_file),
        ]
    )
    logging.info("Subprocess exited")


if __name__ == "__main__":
    base_path = "../../"
    config_path = "config.json"
    worker_path = "workers/"
    iteration = 200000 
    raw_file = base_path + "data/oblique.zarr"
    raw_dataset = "raw/s0"
    out_file = raw_file
    num_workers = 3
    num_cache_workers = 1

    config = {
        "config_path": config_path,
        "worker_path": worker_path,
        "iteration": iteration,
        "raw_file": raw_file,
        "raw_dataset": raw_dataset,
        "out_file": out_file,
        "num_workers": num_workers,
        "num_cache_workers": num_cache_workers,
    }

    start = time.time()

    predict_blockwise(config)

    end = time.time()

    seconds = end - start
    logging.info("Total time to predict: %f seconds" % seconds)
