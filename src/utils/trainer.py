import chainer
import chainer.links as L

from src import settings
from src import models
from src.training.preprocess import ImageNetPreprocess
from src.training.extensions import snapshot_object_to_logdir

LOG_NAME = 'log'
SNAPSHOT_NAME = 'snapshot.npz'


def setup_trainer(
        network,
        dataset,
        lr,
        epoch,
        batch,
        gpu,
        timestamp,
        validation_dataset,
        initialize=False,
):
    optimizer = chainer.optimizers.Adam(lr)
    model = _setup_model(network, gpu, optimizer, initialize)
    extentions = []
    reports = ['epoch']

    train_data = ImageNetPreprocess(dataset)
    train_iter = chainer.iterators.MultithreadIterator(train_data, batch, repeat=True, shuffle=True)
    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer, device=gpu)
    reports.append('main/loss')
    reports.append('main/accuracy')

    val_data = ImageNetPreprocess(validation_dataset)
    val_iter = chainer.iterators.MultithreadIterator(val_data, batch, repeat=False, shuffle=False)
    extentions.append((chainer.training.extensions.Evaluator(val_iter, model, device=gpu), (1, 'epoch')))
    snapshot_trigger_key = 'validation/main/loss'
    reports.append(snapshot_trigger_key)
    reports.append('validation/main/accuracy')
    reports.append('elapsed_time')

    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=settings.TRAIN_LOG_ROOT)

    out, log_name, snapshot_name = _setup_log(timestamp)
    snapshot_trigger = chainer.training.triggers.MinValueTrigger(snapshot_trigger_key, trigger=(1, 'epoch'))
    extentions.extend([
        (chainer.training.extensions.LogReport(log_name=log_name), (1, 'epoch')),
        (chainer.training.extensions.PrintReport(reports), None),
        (chainer.training.extensions.ProgressBar(update_interval=1), None),
        (snapshot_object_to_logdir(model, filename=snapshot_name), snapshot_trigger),
    ])
    [trainer.extend(e, trigger=t) for e, t in extentions]

    return trainer


def _setup_log(timestamp):
    out = settings.TRAIN_LOG_ROOT / timestamp
    out.mkdir(parents=True, exist_ok=True)
    log = str(out / LOG_NAME)
    snapshot = str(out / SNAPSHOT_NAME)
    return out, log, snapshot


def _setup_model(network, gpu, optimizer, initialize):
    model = L.Classifier(models.ARCHS[network]())
    if gpu > -1:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu

    optimizer.setup(model)

    if initialize:
        if network in settings.INITIALIZERS.keys():
            print(f'Initialize Model: {settings.INITIALIZERS[network]}')
            chainer.serializers.load_npz(settings.INITIALIZERS[network], model)
    return model
