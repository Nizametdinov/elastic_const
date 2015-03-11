import elastic_const as ec
import unittest
import tempfile
import os

class TestFemSimulation(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        example_file = os.path.join(dirname, 'test_data\\output_example.txt')
        command = ['cmd', '/C', 'type', example_file]
        self.subject = ec.FemSimulation(command, dirname)
        f, self.config_file = tempfile.mkstemp()
        self.subject.config_file = self.config_file
        os.close(f)

    def test_compute_forces(self):
        forces = self.subject.compute_forces(0, 1, 2, 3, 4, 5)
        self.assertEqual(forces.f1x, -0.17074827)
        self.assertEqual(forces.f1y, -0.01507202)
        self.assertEqual(forces.f2x, -0.17909645)
        self.assertEqual(forces.f2y, -1.93233496)
        self.assertEqual(forces.f3x, 0.34981181)
        self.assertEqual(forces.f3y, 1.94751238)

        conf = None
        with open(self.config_file, encoding = ec.FILE_ENCODING) as conf_file:
            conf = list(map(str.strip, conf_file.readlines()))
        self.assertEqual(len(conf), 6)
        self.assertIn('x1 = 0', conf)
        self.assertIn('y1 = 1', conf)
        self.assertIn('x2 = 2', conf)
        self.assertIn('y2 = 3', conf)
        self.assertIn('x3 = 4', conf)
        self.assertIn('y3 = 5', conf)

    def tearDown(self):
        os.remove(self.config_file)

if __name__ == "__main__":
    unittest.main()