from typing import Generic, TypeVar
import numpy as np
import torch


DType = TypeVar('DType')
Shape = TypeVar('Shape')
Device = TypeVar('Device')


class NumpyArray(np.ndarray, Generic[Shape, DType]):
    ...


class TorchTensor(torch.Tensor, Generic[Shape, DType, Device]):
    ...


# class JaxArray