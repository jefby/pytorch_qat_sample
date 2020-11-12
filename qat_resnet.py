import torch
from torch import quantization

from torchvision import models
qat_resnet18 = models.resnet18(pretrained=True).eval().cuda()

quantization_type = "per_tensor"
# 1. per-tensor quantization
if (quantization_type == "per_tensor"):
    qat_resnet18.qconfig = quantization.QConfig(activation=quantization.default_fake_quant, weight=quantization.default_fake_quant)
else:
    qat_resnet18.qconfig = quantization.QConfig(activation=quantization.default_fake_quant, weight=quantization.default_per_channel_weight_fake_quant)
quantization.prepare_qat(qat_resnet18, inplace=True)
qat_resnet18.apply(quantization.enable_observer)
qat_resnet18.apply(quantization.enable_fake_quant)

dummy_input = torch.randn(1, 3, 224, 224).cuda()
_ = qat_resnet18(dummy_input)
for module in qat_resnet18.modules():
    if isinstance(module, quantization.FakeQuantize):
        module.calculate_qparams()
qat_resnet18.apply(quantization.disable_observer)

qat_resnet18.cuda()


# enable_onnx_checker needs to be disabled because ONNX runtime doesn't support opset 13 yet
#torch.onnx.export(qat_resnet18, dummy_input, "resnet18_qat.onnx", verbose=True, opset_version=13, enable_onnx_checker=False)
if (quantization_type == "per_tensor"):
    torch.onnx.export(qat_resnet18, dummy_input, "resnet18_qat.onnx", verbose=True, opset_version=10, enable_onnx_checker=False)
else:
    torch.onnx.export(qat_resnet18, dummy_input, "resnet18_qat.onnx", verbose=True, opset_version=13, enable_onnx_checker=False)

