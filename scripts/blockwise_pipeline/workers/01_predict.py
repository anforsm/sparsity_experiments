from model import Model
from funlib.persistence import prepare_ds
from funlib.geometry import Roi, Coordinate
import gunpowder as gp
import json
import logging
import math
import numpy as np
import os
import sys
import torch
import zarr
import daisy

from typing import TypedDict

class WorkerConfig(TypedDict):
  iteration: int # The update step of the corresponding model checkpoint to use
  raw_file: str # The .zarr file which contains the raw dataset
  raw_dataset: str # The name of the dataset
  out_file: str # The output file (zarr + dataset) path
  num_cache_workers: int # idk?

logging.basicConfig(level=logging.INFO)
config_path = "config.json"
checkpoint_path = "../../data/checkpoints/"

def predict(config: WorkerConfig):
    iteration = config["iteration"]
    raw_file = config["raw_file"]
    raw_dataset = config["raw_dataset"]
    out_file = config["out_file"]
    num_cache_workers = config["num_cache_workers"]
    
    out_affs_dataset = "test_200000_from_raw/s0"

    # load net config
    with open(os.path.join(config_path)) as f:
        logging.info(
            "Reading setup config from %s" % config_path 
        )
        net_config = json.load(f)

    outputs = net_config["outputs"]
    shape_increase = net_config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,net_config["input_shape"])]
    output_shape = [x + y for x,y in zip(shape_increase,net_config["output_shape"])]

    voxel_size = Coordinate(zarr.open(raw_file,"r")[raw_dataset].attrs["resolution"])
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) / 2

    model = Model()
    model.eval()

    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_affs, output_size)

    logging.log(logging.INFO, f"{raw_file}/{raw_dataset}")
    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_output_roi = source.spec[raw].roi
        total_input_roi = total_output_roi.grow(context, context)

    try:
        device_id = int(daisy.Context.from_env()['worker_id']) % torch.cuda.device_count()
    except Exception as e:
        print(e)
        device_id = 0
    
    predict = gp.torch.Predict(
        model,
        checkpoint=os.path.join(checkpoint_path,f'model_checkpoint_{iteration}'),
        inputs={"input": raw},
        outputs={
            0: pred_affs,
        },
        device=f"cuda:{str(device_id)}"
    )

    write = gp.ZarrWrite(
        dataset_names={
            pred_affs: out_affs_dataset,
        },
        output_filename=out_file,
    )

    scan = gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: 'read_roi',
                pred_affs: 'write_roi',
            },
            num_workers = num_cache_workers)

    pipeline = (
        source
        + gp.Normalize(raw)
        + gp.Pad(raw, None)
        + gp.IntensityScaleShift(raw, 2, -1)
        + gp.Unsqueeze([raw])
        + gp.Unsqueeze([raw])
        + predict
        + gp.Squeeze([pred_affs])
        + gp.IntensityScaleShift(pred_affs, 255, 0)
        + write
        + scan
    )

    predict_request = gp.BatchRequest()

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)
    
    #return total_output_roi


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        run_config = json.load(f)
        
    predict(run_config)