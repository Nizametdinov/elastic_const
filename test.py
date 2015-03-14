import elastic_const as ec
import unittest
import tempfile
import os

dirname = os.path.dirname(os.path.abspath(__file__))

class TestFemSimulation(unittest.TestCase):
    def setUp(self):
        example_file = os.path.join(dirname, 'test_data/output_example.txt')
        if os.name == 'nt':
            command = ['cmd', '/C', 'type', example_file]
        else:
            command = ['cat', example_file]
            ec.PROC_ENCODING = 'utf-16'
        self.subject = ec.FemSimulation(command, dirname)

        f, self.config_file = tempfile.mkstemp()
        self.subject.config_file = self.config_file
        os.close(f)

        self.subject.cache.cache_file_path = os.devnull

    def test_compute_forces(self):
        # It reads from STDOUT computed forces
        forces = self.subject.compute_forces(0, 1, 2, 3, 4, 5)
        self.assertEqual(forces.f1x, -0.17074827)
        self.assertEqual(forces.f1y, -0.01507202)
        self.assertEqual(forces.f2x, -0.17909645)
        self.assertEqual(forces.f2y, -1.93233496)
        self.assertEqual(forces.f3x, 0.34981181)
        self.assertEqual(forces.f3y, 1.94751238)

        # It writes to config file positions of particles
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

        # It caches computed forces
        cached = self.subject.cache.read([0, 1, 2, 3, 4, 5])
        self.assertEqual(cached, forces)

    def tearDown(self):
        os.remove(self.config_file)


class TestForceCache(unittest.TestCase):
    def setUp(self):
        f, self.cache_file = tempfile.mkstemp()
        test_data = """
            0.0 0.0 1.0 1.0e-1 1.0 0.0 -1.707e-1 -1.507e-2 -1.9323 -0.53 0.1 0.2
            0.0 0.0 1.0 2.0e-1 1.0 1.0 -1.007e-1 -1.107e-2 -1.012 -0.53 0.1 0.2
        """
        with open(f, 'w') as cache_file:
            cache_file.write(test_data)
        self.subject = ec.ForceCache(None, cache_file = self.cache_file)

    def test_restore_cache(self):
        self.assertEqual(len(self.subject.values), 2)
        self.assertEqual(
            self.subject.values[0].positions,
            [0.0, 0.0, 1.0, 0.1, 1.0, 0.0]
        )
        self.assertEqual(
            self.subject.values[0].forces(),
            [-0.1707, -0.01507, -1.9323, -0.53, 0.1, 0.2]
        )
        self.assertEqual(self.subject.values[1].f1x, -1.007e-1)
        self.assertEqual(self.subject.values[1].f3y, 0.2)

    def test_save_result(self):
        forces = ec.TripletForces([0, 1, 2, 3, 4, 5], [1.1, 2, 3, 4, 5, 6.1])
        self.subject.save_result(forces)
        self.assertIn(forces, self.subject.values)

        with open(self.cache_file) as cache_file:
            lines = [line.strip() for line in cache_file if line.strip()]
            self.assertEqual(len(lines), 3)
            self.assertEqual(ec.TripletForces.from_string(lines[-1]), forces)

    def test_read(self):
        cached = self.subject.read([0.0, 0.0, 1.0, 0.1, 1.0, 0.0])
        self.assertEqual(cached.positions, [0.0, 0.0, 1.0, 0.1, 1.0, 0.0])
        self.assertEqual(cached.forces(), [-0.1707, -0.01507, -1.9323, -0.53, 0.1, 0.2])

        forces = ec.TripletForces([0, 1, 2, 3, 4, 5], [1.1, 2, 3, 4, 5, 6.1])
        self.subject.save_result(forces)
        cached = self.subject.read(forces.positions)
        self.assertEqual(cached, forces)

    def tearDown(self):
        os.remove(self.cache_file)


if __name__ == "__main__":
    unittest.main()
