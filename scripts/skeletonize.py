from funlib.persistence import open_ds, prepare_ds
from funlib.persistence.graphs import SQLiteGraphDataBase
from funlib.evaluate import detection_scores
import numpy as np
import kimimaro
import networkx as nx
from pathlib import Path
import sys

def convert_to_nx(skels,roi):
  G = nx.Graph()
  node_offset = 0

  offset = roi.offset

  for skel in skels:

    skeleton = skels[skel]

    # Add nodes
    for vertex in skeleton.vertices:
      G.add_node(
        node_offset, 
        id=skeleton.id, 
        position_z=vertex[0]+offset[0], 
        position_y=vertex[1]+offset[1], 
        position_x=vertex[2]+offset[2])

      node_offset += 1
    
    # Add edges
    for edge in skeleton.edges:
      adjusted_u = int(edge[0] + node_offset - len(skeleton.vertices))
      adjusted_v = int(edge[1] + node_offset - len(skeleton.vertices))
      G.add_edge(adjusted_u, adjusted_v, u=adjusted_u, v=adjusted_v)

  return G

def convert_graph_to_sqlite(graph: nx.Graph, path):
  db = SQLiteGraphDataBase(
    Path(path),
    ["position_z", "position_y", "position_x"],
    node_attrs={"label_id": int},
    mode="w",
  )
  for node in graph.nodes.values():
    node["label_id"] = node["id"]
  db.write_graph(graph)
  return db

if __name__ == "__main__":
  defaults = sys.argv[1] == "default"

  if defaults:
    label_file = "../data/oblique.zarr"
    label_ds_name = "labels/s0"
    seg_file = "../data/seg.zarr"
    seg_ds_name = "seg"

  graphml_output = "../data/oblique.graphml"
  sqlite_output = "../data/oblique.sqlite"

  labels = open_ds(label_file, label_ds_name)
  seg = open_ds(seg_file, seg_ds_name)
  roi = seg.roi.intersect(labels.roi)

  labels_arr = labels.to_ndarray(roi)
  seg_arr = seg.to_ndarray(roi)
  vs = tuple(labels.voxel_size)

  teasar_params = {
    "scale": 1.5,
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500,
    "soma_detection_threshold": 750,
    "soma_invalidation_const": 300,
    "soma_invalidation_scale": 2,
    "max_paths": 300,
  }
  
  skels = kimimaro.skeletonize(
    labels_arr, 
    teasar_params, 
    dust_threshold=400,
    anisotropy=vs,
    fix_branching=True,
    fix_borders=True,
    fill_holes=False,
    fix_avocados=False,
    progress=True,
    #parallel=10,
    #parallel_chunk_size=10,
  )

  uniques = np.unique(labels_arr)
  uniques = uniques[uniques > 0]
  skel_ids = np.array(list(skels.keys()))
  assert all(np.isin(uniques, skel_ids))

  G = convert_to_nx(skels, roi)

  nx.write_graphml(G, graphml_output)

  convert_graph_to_sqlite(G, sqlite_output)