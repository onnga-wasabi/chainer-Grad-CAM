import chainer
import chainer.links as L


class ImageNetVGG16(L.VGG16Layers):

    def __init__(self):
        super(ImageNetVGG16, self).__init__()

    def prepare_cam(self, x):
        with chainer.no_backprop_mode():
            h = self(x, layers=['conv5_3'])['conv5_3']
        self.target = h
        for funcs in [func for key, func in self.functions.items() if key in ['pool5', 'fc6', 'fc7', 'fc8']]:
            for f in funcs:
                h = f(h)
        return h
