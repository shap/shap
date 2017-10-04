import unittest
import iml
import numpy as np
import scipy
import itertools
import math
import copy

class TestIML(unittest.TestCase):

    def test_ESExplainer_basic(self):
        # The esvalues package contains most of the tests, we just make sure the iml interface is good
        N = 1
        P = 10
        X = np.random.randn(N,P)
        f = lambda x: np.sum(x,1)
        x = np.random.randn(1,P)
        e = iml.ESExplainer(f, X).explain(x)
        self.assertTrue(e.baseValue + np.sum(e.effects) - f(x)[0] < 1e-6)
        self.assertTrue(e.baseValue - f(X)[0] < 1e-6)
        for i in range(P):
            self.assertTrue(x[0,i]-X[0,i] - e.effects[i] < 1e-6)

    def test_ESExplainer_groups(self):
        N = 1
        P = 10
        X = np.random.randn(N,P)
        f = lambda x: np.sum(x,1)
        x = np.random.randn(1,P)
        e = iml.ESExplainer(f, X, featureGroups=[np.array([0,1,2,3,4,5]), np.array([6]), np.array([7,8,9])]).explain(x)
        self.assertTrue(e.baseValue + np.sum(e.effects) - f(x)[0] < 1e-6)
        self.assertTrue(e.baseValue - f(X)[0] < 1e-6)
        self.assertTrue(np.sum(x[0,0:6]-X[0,0:6]) - e.effects[0] < 1e-6)
        self.assertTrue(np.sum(x[0,6]-X[0,6]) - e.effects[1] < 1e-6)
        self.assertTrue(np.sum(x[0,7:]-X[0,7:]) - e.effects[2] < 1e-6)
