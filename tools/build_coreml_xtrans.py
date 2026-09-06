"""Build the CoreML X-Trans variant with correctly rounded float32 division.

The original graph remains the reference and is used by other providers.
CoreML GPU division perturbs near-zero direction scores and changes branch
decisions. Evaluate the 16 float32 divisions by three through float64 on the CPU EP, then
round back to float32; other supported operations remain eligible for CoreML.
MIGraphX additionally requires all float divisions and explicit x*x squares,
plus strict compiler settings in its isolated child. No coefficients,
comparisons, image boundaries, or tolerance are changed.
"""
import argparse
import hashlib
from pathlib import Path

import onnx

SOURCE_SHA256 = "d22747f383d898715541dfbf68d20b21c632179b152e9c8a52f3025116b32d40"


def build(source: Path, destination: Path, *, backend="coreml"):
    if backend not in {"coreml", "migraphx"}:
        raise ValueError("Unsupported precision backend")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("X-Trans source graph changed; revalidate the CoreML variant")
    model = onnx.load_model_from_string(data)
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
    constants = {
        node.output[0]: onnx.numpy_helper.to_array(node.attribute[0].t)
        for node in model.graph.node if node.op_type == "Constant"
    }
    nodes = []
    count = 0
    for node in model.graph.node:
        if (node.op_type == "Div"
                and types.get(node.output[0]) == onnx.TensorProto.FLOAT
                and (backend == "migraphx" or (node.input[1] in constants
                     and bool((constants[node.input[1]] == 3).all())))):
            for index, name in enumerate(node.input):
                cast = name + "_div_double_" + node.name.replace("/", "")
                nodes.append(onnx.helper.make_node(
                    "Cast", [name], [cast], to=onnx.TensorProto.DOUBLE,
                ))
                node.input[index] = cast
            original = node.output[0]
            node.output[0] = original + "_precise_div"
            nodes.append(node)
            nodes.append(onnx.helper.make_node(
                "Cast", [node.output[0]], [original], to=onnx.TensorProto.FLOAT,
            ))
            count += 1
        elif (backend == "migraphx" and node.op_type == "Pow"
              and node.input[1] in constants
              and bool((constants[node.input[1]] == 2).all())):
            # GPU pow(x, 2) is not necessarily the correctly rounded x*x.
            nodes.append(onnx.helper.make_node(
                "Mul", [node.input[0], node.input[0]], list(node.output),
                name=node.name + "_square",
            ))
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--backend", choices=("coreml", "migraphx"), default="coreml")
    args = parser.parse_args()
    print(f"Preserved division precision for {build(args.source, args.destination, backend=args.backend)} nodes")