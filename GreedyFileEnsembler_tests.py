import unittest
from GreedyFileEnsembler4 import *

class T(unittest.TestCase):
  def test_GreedyFileEnsembler4_max_replacements_0(self):
    g = GreedyFileEnsembler4(sklearn.metrics.r2_score, 10, max_replacements=0)
    trains = [pd.Series(np.random.random(100)) for i in range(20)]
    y = pd.Series(np.random.random(100))
    g.fit(trains, y)
    indexes = g.indexes
    self.assertEqual(10, len(indexes))
    self.assertEqual(10, len(np.unique(indexes)))

  def test_GreedyFileEnsembler4_max_replacements_1(self):
    g = GreedyFileEnsembler4(sklearn.metrics.r2_score, 10, max_replacements=1)
    trains = [pd.Series(np.random.random(100)) for i in range(20)]
    y = pd.Series(np.random.random(100))
    g.fit(trains, y)
    indexes = g.indexes
    self.assertEqual(10, len(indexes))
    uniques = np.unique(indexes)
    for u in uniques:
      self.assertTrue(indexes.count(u) <= 2)

  def test_GreedyFileEnsembler4_max_replacements_0_not_enough(self):
    g = GreedyFileEnsembler4(sklearn.metrics.r2_score, 20, max_replacements=0)
    trains = [pd.Series(np.random.random(100)) for i in range(10)]
    y = pd.Series(np.random.random(100))
    g.fit(trains, y)
    indexes = g.indexes
    self.assertEqual(10, len(indexes))
    self.assertEqual(10, len(np.unique(indexes)))
    # will exit early