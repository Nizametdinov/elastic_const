from elastic_const import forces as fs
import unittest
import tempfile
import os
import math
import numpy as np
import numpy.testing as np_test
import io

dirname = os.path.dirname(os.path.abspath(__file__))


class TestPairForce(unittest.TestCase):
    def setUp(self):
        pass

    def test_rotate(self):
        force = fs.PairForce(distance=2 * math.sqrt(2), force=4. * math.sqrt(2))
        force_x, force_y = force.rotate([2., -2.], origin=[0., 0.])
        np_test.assert_almost_equal(force_x, 4.)
        np_test.assert_almost_equal(force_y, -4.)

        force_x, force_y = force.rotate([2. * math.sqrt(2), -2.], origin=[0., -2.])
        np_test.assert_almost_equal(force_x, 4. * math.sqrt(2))
        np_test.assert_almost_equal(force_y, 0.)


class TestTripletFemSimulation(unittest.TestCase):
    def setUp(self):
        example_file = os.path.join(dirname, 'test_data/output_example.txt')
        if os.name == 'nt':
            command = ['cmd', '/C', 'type', os.path.normpath(example_file)]
        else:
            command = ['cat', example_file]
            fs.PROC_ENCODING = 'utf-16'
        self.subject = fs.TripletFemSimulation(command, dirname)

        f, self.config_file = tempfile.mkstemp()
        self.subject.config_file = self.config_file
        os.close(f)

        self.subject.cache.cache_file_path = os.devnull

    def test_compute_forces(self):
        # It reads from STDOUT computed forces
        forces = self.subject.compute_forces(np.array([[0, 1], [2, 3], [4, 5]]))
        self.assertEqual(forces.force(1, 'x'), -0.17074827)
        self.assertEqual(forces.force(1, 'y'), -0.01507202)
        self.assertEqual(forces.force(2, 'x'), -0.17909645)
        self.assertEqual(forces.force(2, 'y'), -1.93233496)
        np_test.assert_equal(forces.force(3), np.array([0.34981181, 1.94751238]))

    def test_writes_to_config_file_positions_of_particles(self):
        self.subject.compute_forces(np.array([[0, 1], [2, 3], [4, 5]]))
        with open(self.config_file, encoding=fs.FILE_ENCODING) as conf_file:
            conf = list(map(str.strip, conf_file.readlines()))
        self.assertEqual(len(conf), 6)
        self.assertIn('x1 = 0', conf)
        self.assertIn('y1 = 1', conf)
        self.assertIn('x2 = 2', conf)
        self.assertIn('y2 = 3', conf)
        self.assertIn('x3 = 4', conf)
        self.assertIn('y3 = 5', conf)

    def test_caches_computed_forces(self):
        computed = self.subject.compute_forces(np.array([[0, 1], [2, 3], [4, 5]]))
        cached = self.subject.cache.read(np.array([[0, 1], [2, 3], [4, 5]]))
        self.assertEqual(cached, computed)

    def test_reads_cached_value(self):
        cached = fs.TripletForces(np.array([[0, 1], [2, 3], [4, 5]]), [1.1, 2, 3, 4, 5, 6.1])
        self.subject.cache.save_result(cached)

        computed = self.subject.compute_forces(np.array([[0, 1], [2, 3], [4, 5]]))
        self.assertEqual(cached, computed)

    def test_plan_generation(self):
        plan_file = io.StringIO()
        self.subject = fs.TripletFemSimulation(self.subject.command, dirname, plan_file=plan_file)

        # It returns zeroes
        forces = self.subject.compute_forces(np.array([[0., 1.], [2., 3.], [4., 5.]]))
        np_test.assert_equal(forces.force(1), np.array([0., 0.]))
        np_test.assert_equal(forces.force(2), np.array([0., 0.]))
        np_test.assert_equal(forces.force(3), np.array([0., 0.]))

        # It generates computation plan
        self.subject.compute_forces(np.array([[0., 0.], [1., 1.], [2., 2.]]))
        plan = plan_file.getvalue().splitlines()
        self.assertEqual(list(map(float, plan[0].split())), [0., 1., 2., 3., 4., 5.])
        self.assertEqual(list(map(float, plan[1].split())), [0., 0., 1., 1., 2., 2.])

        # It does not output the same configuration twice
        self.subject.compute_forces(np.array([[0., 0.], [1., 1.], [2., 2.]]))
        plan = plan_file.getvalue().splitlines()
        self.assertEqual(len(plan), 2)

    def tearDown(self):
        os.remove(self.config_file)


class TestTripletForceCache(unittest.TestCase):
    def setUp(self):
        f, self.cache_file = tempfile.mkstemp()
        test_data = """
            0.0 0.0 1.0 1.0e-1 1.0 0.0 -1.707e-1 -1.507e-2 -1.9323 -0.53 0.1 0.2
            0.0 0.0 1.0 2.0e-1 1.0 1.0 -1.007e-1 -1.107e-2 -1.012 -0.53 0.1 0.2
        """
        with open(f, 'w') as cache_file:
            cache_file.write(test_data)
        self.subject = fs.TripletForceCache(None, cache_file=self.cache_file)

    def test_restore_cache(self):
        normalized_force = fs.TripletForces(
            np.array([[0.0, 0.0], [1.0, 0.1], [1.0, 0.0]]),
            [-0.1707, -0.01507, -1.9323, -0.53, 0.1, 0.2]
        ).normalized()
        self.assertEqual(len(self.subject.values), 2)
        np_test.assert_equal(
            self.subject.values[0].positions,
            normalized_force.positions
        )
        self.assertEqual(
            self.subject.values[0].forces,
            normalized_force.forces
        )

    def test_save_result(self):
        normalized_force = fs.TripletForces([0, 0, 1, 0, 1, 2], [1.1, 2, 3, 4, 5, 6.1])
        self.subject.save_result(normalized_force)
        self.assertIn(normalized_force, self.subject.values)

        with open(self.cache_file) as cache_file:
            lines = [line.strip() for line in cache_file if line.strip()]
            self.assertEqual(len(lines), 3)
            self.assertEqual(fs.TripletForces.from_string(lines[-1]), normalized_force)

    def test_read(self):
        cached = self.subject.read(np.array([[0.0, 0.0], [1.0, 0.1], [1.0, 0.0]]))
        np_test.assert_equal(cached.positions, [np.array([0.0, 0.0]), np.array([1.0, 0.1]), np.array([1.0, 0.0])])
        self.assertEqual(cached.forces, [-0.1707, -0.01507, -1.9323, -0.53, 0.1, 0.2])

        forces = fs.TripletForces([1, 1, 2, 3, 4, 5], [1.1, 2, 3, 4, 5, 6.1])
        self.subject.save_result(forces)
        cached = self.subject.read(np.array([[1, 1], [2, 3], [4, 5]]))
        self.assertEqual(cached, forces)

        # compares float with tolerance
        cached = self.subject.read(np.array([[1, 0.7 + 0.2 + 0.1], [2, (0.1 + 0.2) * 10], [4, 5]]))
        self.assertEqual(cached, forces)

    def test_read_with_xy_symmetry(self):
        forces = fs.TripletForces([0, 0, 1, 2, 2, 0], [-1, -0.5, 0., 1., 1., -0.5])
        self.subject.save_result(forces)

        cached = self.subject.read(np.array([[0, 0], [2, 1], [0, 2]]))
        self.assertIsNotNone(cached)
        np_test.assert_equal(cached.positions, np.array([[0, 0], [2, 1], [0, 2]]))
        self.assertEqual(cached.forces, [-0.5, -1, 1., 0., -0.5, 1.])

    def test_read_with_shift(self):
        forces = fs.TripletForces([0, 0, 4, 0, 4, 3], [1, 2, 3, 4, 5, 6])
        self.subject.save_result(forces)

        cached = self.subject.read(np.array([[0, 0], [0, 3], [-4, 0]]))
        self.assertIsNotNone(cached)
        np_test.assert_equal(cached.positions, np.array([[0, 0], [0, 3], [-4, 0]]))
        self.assertEqual(cached.forces, [3, 4, 5, 6, 1, 2])

    def test_read_with_rotation(self):
        forces = fs.TripletForces([0, 0, 4, 0, 4, 3], [1, 2, 3, 4, 5, 6])
        self.subject.save_result(forces)

        cached = self.subject.read(np.array([[0, 0], [0, 4], [-3, 4]]))
        self.assertIsNotNone(cached)
        np_test.assert_equal(cached.positions, np.array([[0, 0], [0, 4], [-3, 4]]))
        self.assertEqual(cached.forces, [-2, 1, -4, 3, -6, 5])

    def test_read_with_rotation_and_mirror(self):
        forces = fs.TripletForces(np.array([[0, 0], [0, 1], [-2, 1]]), [0.3, -1.2, 0.5, 1., -0.8, 0.2])
        self.subject.save_result(forces)

        cached = self.subject.read(np.array([[0, 0], [-1, 0], [-1, 2]]))
        self.assertIsNotNone(cached)
        np_test.assert_equal(cached.positions, np.array([[0, 0], [-1, 0], [-1, 2]]))
        self.assertEqual(cached.forces, [1.2, -0.3, -1., -0.5, -0.2, 0.8])

    def tearDown(self):
        os.remove(self.cache_file)


class TestPairFemSimulation(unittest.TestCase):
    def setUp(self):
        example_file = os.path.join(dirname, 'test_data/2particles_example_output.txt')
        if os.name == 'nt':
            command = ['cmd', '/C', 'type', os.path.normpath(example_file)]
        else:
            command = ['cat', example_file]
            fs.PROC_ENCODING = 'utf-16'
        self.subject = fs.PairFemSimulation(command, dirname)

        f, self.config_file = tempfile.mkstemp()
        self.subject.config_file = self.config_file
        os.close(f)

        self.subject.cache.cache_file_path = os.devnull

    def test_compute_forces(self):
        # It reads from STDOUT computed forces
        force = self.subject.compute_forces(3.)
        self.assertEqual(force.force, 2.727838374946)
        self.assertEqual(force.distance, 3.)

    def test_writes_to_config_file_positions_of_particles(self):
        forces = self.subject.compute_forces(2)
        conf = None
        with open(self.config_file, encoding=fs.FILE_ENCODING) as conf_file:
            conf = list(map(str.strip, conf_file.readlines()))
        self.assertEqual(len(conf), 1)
        self.assertIn('distance = 2', conf)

    def test_caches_computed_forces(self):
        computed = self.subject.compute_forces(4)
        cached = self.subject.cache.read(4)
        self.assertIsNotNone(cached)
        self.assertEqual(cached, computed)

    def test_reads_cached_value(self):
        cached = fs.PairForce(3, 1.1)
        self.subject.cache.save_result(cached)

        computed = self.subject.compute_forces(3)
        self.assertEqual(cached, computed)

    def test_plan_generation(self):
        plan_file = io.StringIO()
        self.subject = fs.PairFemSimulation(self.subject.command, dirname, plan_file=plan_file)

        # It returns zeroes
        force = self.subject.compute_forces(3)
        self.assertEqual(force.force, 0.)
        self.assertEqual(force.distance, 3.)

        # It generates computation plan
        self.subject.compute_forces(2.5)
        plan = plan_file.getvalue().splitlines()
        self.assertEqual(float(plan[0]), 3.)
        self.assertEqual(float(plan[1]), 2.5)

        # It does not output the same configuration twice
        self.subject.compute_forces(2.5)
        plan = plan_file.getvalue().splitlines()
        self.assertEqual(len(plan), 2)

    def tearDown(self):
        os.remove(self.config_file)


if __name__ == "__main__":
    unittest.main()
