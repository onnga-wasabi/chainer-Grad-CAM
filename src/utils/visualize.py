import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import cv2

from src.utils.datasets import ImageNetClassID


def get_heatmap(model, image, classids, rank=0):
    with chainer.using_config('train', False):
        pred = model.prepare_cam(image)
    target_class = np.argsort(F.softmax(pred)[0].array)[::-1][rank]
    signal = np.zeros_like(pred).astype('f')
    signal[0, target_class] = 1
    pred.grad = signal
    pred.backward()

    map_weights = _global_average_pooling_2d(model.target.grad)
    activations = model.target

    grad_cams = map_weights * activations
    grad_cam = F.relu(F.sum(grad_cams, axis=1))[0]
    grad_cam = cv2.resize(grad_cam.array, image.shape[2:])
    heatmap = (grad_cam / np.max(grad_cam)) * 255
    heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)[:, :, ::-1]

    return heatmap, classids[target_class]


def get_heatmap_imagenet(model, image, classids=ImageNetClassID, rank=0):
    x = L.model.vision.vgg.prepare(image)
    x = chainer.Variable(np.array([x]).astype('f'))
    return get_heatmap(model, x, classids=ImageNetClassID, rank=0)


def _global_average_pooling_2d(maps):
    return F.average_pooling_2d(maps, maps.shape[2:])
