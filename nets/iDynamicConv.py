from collections import namedtuple
from string import Template

import cupy  # idynamic implement is based on cupy-cuda
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

Stream = namedtuple("Stream", ["ptr"])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return "float"
    elif isinstance(t, torch.cuda.DoubleTensor):
        return "double"


class _idynamic(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel(
                "idynamic_forward_kernel",
                _idynamic_kernel,
                Dtype=Dtype(input),
                nthreads=n,
                num=batch_size,
                channels=channels,
                groups=weight.size()[1],
                bottom_height=height,
                bottom_width=width,
                top_height=output_h,
                top_width=output_w,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                stride_h=stride[0],
                stride_w=stride[1],
                dilation_h=dilation[0],
                dilation_w=dilation[1],
                pad_h=padding[0],
                pad_w=padding[1],
            )
            f(block=(CUDA_NUM_THREADS, 1, 1), grid=(GET_BLOCKS(n), 1, 1), args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()], stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(
            Dtype=Dtype(grad_output),
            num=batch_size,
            channels=channels,
            groups=weight.size()[1],
            bottom_height=height,
            bottom_width=width,
            top_height=output_h,
            top_width=output_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride[0],
            stride_w=stride[1],
            dilation_h=dilation[0],
            dilation_w=dilation[1],
            pad_h=padding[0],
            pad_w=padding[1],
        )

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt["nthreads"] = n

                f = load_kernel("idynamic_backward_grad_input_kernel", _idynamic_kernel_backward_grad_input, **opt)
                f(
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(n), 1, 1),
                    args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
                )

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt["nthreads"] = n

                f = load_kernel("idynamic_backward_grad_weight_kernel", _idynamic_kernel_backward_grad_weight, **opt)
                f(
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(n), 1, 1),
                    args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream),
                )

        return grad_input, grad_weight, None, None, None


def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """idynamic kernel"""
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _idynamic.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class IDynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, group_channels, bias=True):
        super(IDynamicDWConv, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=bias),
            nn.Conv2d(channels // reduction_ratio, channels // reduction_ratio, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels // reduction_ratio, bias=bias),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(channels // reduction_ratio, kernel_size**2 * self.groups, 1, bias=bias))

    def forward(self, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _idynamic_cuda(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        return out


if __name__ == "__main__":
    img = torch.randn(16, 128, 56, 56).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = IDynamicDWConv(128, kernel_size=5, group_channels=4).to(device)
    output = net(img)
