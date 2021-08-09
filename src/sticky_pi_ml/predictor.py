from sticky_pi_ml.ml_bundle import BaseMLBundle
from typing import Union


class BasePredictor(object):
    def __init__(self, ml_bundle: BaseMLBundle):
        self._ml_bundle = ml_bundle
        self._name = ml_bundle.name
        self._version = ml_bundle.version

    @property
    def version(self):
        return self._version

    @property
    def name(self):
        return self._name
