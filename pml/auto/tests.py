from __future__ import print_function, absolute_import

import unittest
from introspect_sklearn import *

class TestSequenceFunctions(unittest.TestCase):
  def setUp(self):
    pass

  def test_parse_floats(self):
    def tr(exp, act):
      self.assertEqual(100, len(act))
      self.assertEqual(exp, [act[0], act[-1]])
    pft = parse_float_type
    tr([0., 1.], pft("float, 0 < support_fraction < 1"))
    tr([0., 0.5], pft("float, 0. < contamination < 0.5"))
    tr([-100, 100], pft("float, optional (default=1.)"))
    tr([-100, 100], pft("float, optional"))
    tr([-100, 100], pft("float"))


  def test_parse_range(self):
    prt = parse_range_type
    self.assertEqual(['SAMME', 'SAMME.R'], prt("{'SAMME', 'SAMME.R'}"))
    self.assertEqual(['exponential', 'linear', 'square'], prt("{'linear', 'square', 'exponential'}"))
    self.assertEqual(['deviance'], prt("{'deviance'}"))
    self.assertEqual(['huber', 'lad', 'ls', 'quantile'], prt("{'ls', 'lad', 'huber', 'quantile'}"))
    self.assertEqual([False, True, 'auto'], prt("{True, False, 'auto'}"))
    self.assertEqual([None, 'auto', 'eigen', 'svd'], prt("{None, 'auto', 'svd', eigen'}"))
    self.assertEqual(['auto', 'ball_tree', 'brute', 'kd_tree'], prt("{'auto', 'ball_tree', 'kd_tree', 'brute'}"))
    self.assertEqual(['knn', 'rbf'], prt("{'knn', 'rbf'}"))

  def test_parse_string(self):
    pst = parse_string_type
    self.assertEqual(['entropy', 'gini'], pst("string, optional (default=\"gini\")", "string, optional (default=\"gini\") - The function to measure the quality of a split. Supported criteria are\"gini\" for the\n Gini impurity and \"entropy\" for the information gain."))
    self.assertEqual(['l1', 'l2'], pst("string, 'l1' or 'l2'", "Used to specify the norm used in the penalization."))
    self.assertEqual(['linear', 'poly', 'precomputed', 'rbf', 'sigmoid'], pst("string, optional (default='rbf')", "Specifies the kernel type to be used in the algorithm.It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ora callable.If none is given, 'rbf' will be used. If a callable is given it isused to precompute the kernel matrix."))



if __name__ == '__main__':
  unittest.main()
