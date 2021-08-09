import unittest
from sticky_pi_ml.insect_tuboid_classifier.taxonomy import TaxonomyMapper
import numpy as np

label_yaml = """
LABELS:
  - ['^Background.*', 0]
  - ['^Ambiguous.*', 1]
  - ['^Insecta\.Hemiptera\.Cicadellidae\.Edwardsiana.*', null]
  - ['^Insecta\.Diptera\.Drosophilidae\.Drosophila\.Drosophila suzukii.*', null]
  - ['^Insecta\.Diptera\.Drosophilidae.*', null]
  - ['^Insecta\.Diptera\.Psychodidae.*', null]
  - ['^Insecta\.Diptera\.Culicidae.*', null]
  - ['^Insecta\.Diptera\.Muscidae.*',null]
  - ['^Insecta\.Diptera\.Sciaridae.*', null]
  - ['^Insecta\.Coleoptera\.Curculionidae.*',null]
  - ['^Insecta\.Coleoptera\.Coccinellidae.*',null]
  - ['^Insecta\.Coleoptera.*',null]
  - ['^Insecta\.Hymenoptera\.Figitidae.*', null]
  - ['^Insecta\.Hymenoptera.*',null]
  - ['^Insecta.*',2]
"""

class TestTaxonomy(unittest.TestCase):

    def test_map(self):
        import yaml
        dct = yaml.safe_load(label_yaml)

        tm = TaxonomyMapper(dct['LABELS'])

        self.assertEqual(tm.label_to_level_dict(4)['species'], 'Drosophila suzukii')
