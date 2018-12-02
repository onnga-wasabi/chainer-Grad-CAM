import chainer


def snapshot_object_to_logdir(target, filename, savefun=chainer.serializers.npz.save_npz):
    @chainer.training.extension.make_extension(priority=-100)
    def snapshot_object_to_logdir(trainer):
        _snapshot_object_to_logdir(trainer, target, filename, savefun)

    return snapshot_object_to_logdir


def _snapshot_object_to_logdir(trainer, target, filename, savefun):
    savefun(filename, target)
