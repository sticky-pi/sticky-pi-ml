import itertools
from collections import OrderedDict
import re
from typing import Tuple, Dict, List, Union


class TaxonomyMapper(object):
    _taxonomic_levels = ["type", "order", "family", "genus", "species"]

    def __init__(self, class_label_list: List[Tuple[str, Union[int, None]]]):
        """
        `label' is an unique integer describing the discrete class
        `pattern' is a regex formatted string: type.order.family.genus.species

        :param class_label_list: a list of tuples (pattern, label).
            label can be None, in which case it will be auto allocated -- next available int.
        """

        # re-pattern (str) -> label (int)
        self._pattern_to_label_map = OrderedDict(class_label_list)
        # fill the none values with auto labels (alphabetically)
        auto_label = max([v for v in self._pattern_to_label_map.values() if v is not None]) + 1
        for k in self._pattern_to_label_map:
            if self._pattern_to_label_map[k] is None:
                self._pattern_to_label_map[k] = auto_label
                auto_label += 1

        self._label_to_pattern_map = {}
        for k, v in self._pattern_to_label_map.items():
            assert v not in self._label_to_pattern_map
            self._label_to_pattern_map[v] = k


    @property
    def n_classes(self) -> int:
        return len(self._pattern_to_label_map)

    def level_dict_to_tuple(self, dic: dict):
        t = (dic[rank] for rank in self._taxonomic_levels)
        return tuple(t)

    def level_dict_to_label(self, dic: Dict[str, str]) -> int:
        tupl = (dic[rank] for rank in self._taxonomic_levels)
        return self.tuple_to_label(tupl)

    def tuple_to_label(self, tupl: Tuple) -> int:
        string = ".".join(tupl)
        for k, v in self._pattern_to_label_map.items():
            if re.match(k, string):
                return v
        raise Exception("No matching label for %s" % string)

    def label_to_pattern(self, label: int) -> str:
        return self._label_to_pattern_map[label]

    def label_to_level_dict(self, label: int) -> Dict[str, str]:
        pattern = self.label_to_pattern(label)
        fields = [re.sub('[^a-zA-Z ]', '', f) for f in pattern.split(r'\.')]
        assert 0 < len(fields) <= len(self._taxonomic_levels)
        out = {tax: f for tax, f in itertools.zip_longest(self._taxonomic_levels, fields)}
        return out
