
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import onnxruntime


def torch_to_onnx(torch_model, input_example, output_path):
    torch_model.eval()
    print("input example shape: ", input_example.shape)
    # Input to the model
    x = input_example.requires_grad_()
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      output_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})

    verify_onnx(output_path, torch_out, x)


def verify_onnx(onnx_model_path, torch_out, x):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    all([np.testing.assert_allclose(to_numpy(torch_out[i]), ort_outs[i], rtol=1e-03, atol=1e-05)
        for i in range(len(ort_outs))])

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
