"""Build fixed-tile RCD for MIGraphX without large Tile-generated kernels.

Two axis-wise Gather operations repeat each 2x2 CFA mask exactly. Arithmetic,
coefficients, white balance, matrix and borders remain in the original graph.
This is a build tool only; startup never parses or rewrites ONNX models.
"""
import argparse
import hashlib
from pathlib import Path

import onnx

SOURCE_SHA256 = "d15dfdfa0d0a80646bee2e148cc0e5a07e6b2554008b1b3ee0458951bd8fabf4"
TILE = 1536


def build(source: Path, destination: Path):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("RCD source changed; revalidate the MIGraphX variant")
    model = onnx.load_model_from_string(data)
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "Tile":
            nodes.append(node)
            continue
        count += 1
        name = f"rcd_mask_{count}"
        indices = name + "_indices"
        model.graph.initializer.append(onnx.helper.make_tensor(
            indices, onnx.TensorProto.INT64, [TILE], [i % 2 for i in range(TILE)],
        ))
        nodes.extend([
            onnx.helper.make_node("Gather", [node.input[0], indices], [name + "_rows"],
                                  axis=0, name=name + "_row_gather"),
            onnx.helper.make_node("Gather", [name + "_rows", indices], list(node.output),
                                  axis=1, name=name + "_col_gather"),
        ])
    if count != 3:
        raise ValueError("Expected exactly three CFA mask Tile operations")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Make incompatible inputs fail explicitly, instead of broadcasting masks
    # for one size over a differently-sized RAW tensor.
    for value in list(model.graph.input) + list(model.graph.output):
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_param in {"h", "w"}:
                dimension.dim_value = TILE
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"Replaced {build(args.source, args.destination)} CFA mask Tile operations")
