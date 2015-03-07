import elastic_const as ec
import unittest
from os import path

class TestFemSimulation(unittest.TestCase):
    def setUp(self):
        dirname = path.dirname(path.abspath(__file__))
        example_file = path.join(dirname, 'test_data\\output_example.txt')
        command = ['cmd', '/C', 'type', example_file]
        self.subject = ec.FemSimulation(command, dirname)

    def test_compute_forces(self):
        forces = self.subject.compute_forces(0, 0, 1, 1, 2, 2)
        self.assertEqual(forces, {
            'f1x': -0.17074827, 'f1y': -0.01507202,
            'f2x': -0.17909645, 'f2y': -1.93233496,
            'f3x': 0.34981181, 'f3y': 1.94751238
            })

if __name__ == "__main__":
    unittest.main()