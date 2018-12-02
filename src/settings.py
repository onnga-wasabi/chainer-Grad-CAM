from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPTS_DIR = ROOT / 'src'
INITIALIZERS = SCRIPTS_DIR / 'initializers'
DATA = Path.home() / 'data'
DATA_FOOD = DATA / 'food-101'

# == Format ==
TIME_STAMP_FORMAT = '%y-%m-%d/%H%M%S'

# == Arguments Defaults ==
ARGS_DEFAULTS = {
    'gpu': -1,
    'batch': 32,
    'epoch': 50,
    'network': 'food',  # ARCHS['network'] - brain.models.__init__.py
    'learning_late': 1e-2,
    'init': 'store_true',  # default fase
}

# == Logging ==
TRAIN_LOG_ROOT = ROOT / 'result'  # 権限ないところへ変更するならディレクトリ作成実行権限付与までやってわたして

# == Created Images ==
IMAGE_DIR = ROOT / 'images'
IMAGE_SLICES = IMAGE_DIR / 'slices'  # 再構成画像に関するもの
IMAGE_OTHERS = IMAGE_DIR / 'others'  # 中間表現の分布など再構成画像以外もの
