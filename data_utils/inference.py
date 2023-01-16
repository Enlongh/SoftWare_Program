import numpy as np
from src.config import config

from models.unet3plus import UNet3Plus
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
def load_net(path=None):
    net = UNet3Plus(in_channels=config.slice_stack, n_classes=config.num_classes)
    if path == None:
        path = 'models/best_model.ckpt'
    param_dict = load_checkpoint(path)
    load_param_into_net(net, param_dict)
    return net


def inference(net, data):
    """

    :param net: 用于推理的神经网络
    :param data: 输入img，形状应该为【H，W】或者【D，H，W】
    :return: 推理结果，形状应该为【H，W】或者【D，H，W】
    """
    if len(data.shape) == 2:
        data = data[None][None]  # [1,1,H,W]
        input = Tensor.from_numpy(data)
        input = input.astype("float32")
        input = input / 255.0
        predictions = net(input)  # [1,3,H,W]
        predictions = predictions.asnumpy()
        predictions = np.argmax(predictions, axis=1)  # [1,H,W]
        predictions = predictions[0]  # [H,W]
    elif len(data.shape) == 3:
        data = np.expand_dims(data, axis=1)
        D, _, H, W = data.shape
        bs = config.batch_size
        predictions = np.zeros(shape=(D, H, W))

        for i in range(int(np.ceil(D / bs))):
            input = data[i * bs:min(D, (i + 1) * bs), :, :]
            input = np.expand_dims(input, axis=1)
            input = np.ascontiguousarray(input)
            input = Tensor.from_numpy(input)
            prediction = net(input).asnumpy()  # [bs,num_class,H,W]
            prediction = np.argmax(prediction, axis=1)  # [bs,H,W]
            predictions[i * bs:min(D, (i + 1) * bs), :, :] = prediction[:, :, :]
    else:
        predictions = data
    return predictions
