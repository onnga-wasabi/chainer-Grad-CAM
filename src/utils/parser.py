import argparse

from src.models import ARCHS
from src.settings import ARGS_DEFAULTS


def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, default=ARGS_DEFAULTS['gpu'])
    parser.add_argument('-b', '--batch', type=int, default=ARGS_DEFAULTS['batch'])
    parser.add_argument('-e', '--epoch', type=int, default=ARGS_DEFAULTS['epoch'])
    parser.add_argument('-n', '--network', default=ARGS_DEFAULTS['network'], choices=ARCHS.keys())
    parser.add_argument('--lr', default=ARGS_DEFAULTS['learning_late'])
    parser.add_argument('--init', action=ARGS_DEFAULTS['init'])
    parser.add_argument('--comment')

    return parser.parse_args()
