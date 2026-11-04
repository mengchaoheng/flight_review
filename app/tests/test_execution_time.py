"""Execution-time units, schema selection and native sample times."""

import os
from types import SimpleNamespace
import unittest

import numpy as np
from bokeh.models import LogScale, LogTicker, LogTickFormatter

from config import plot_config
from execution_time import get_execution_time_plot
from helper import load_ulog_file


def topic(name, times, instance=0, **fields):
    return SimpleNamespace(name=name, multi_id=instance,
                           data={'timestamp': np.asarray(times, dtype=np.uint64),
                                 **{key: np.asarray(value) for key, value in fields.items()}})


class ExecutionTimeTest(unittest.TestCase):
    def test_new_fields_and_both_matrices_keep_native_times_and_units(self):
        data = [
            topic('om_mpc_command', [10000, 20000, 30000],
                  mpc_compute_us=[5000, 5200, 5100], qp_solve_us=[4000, 4200, 4100],
                  solve_us=[999, 999, 999], solver_us=[888, 888, 888]),
            topic('control_allocator_diagnostics', [11000, 21000],
                  allocation_running_time=[4, 5]),
            topic('control_allocator_diagnostics', [12000, 22000], instance=1,
                  allocation_running_time=[40, 45]),
            topic('control_allocator_status', [13000], instance=1,
                  allocation_running_time=[999]),
        ]
        plot = get_execution_time_plot(data, plot_config).bokeh_plot
        self.assertEqual(len(plot.renderers), 4)
        for renderer, times, values in zip(plot.renderers,
                [[10000, 20000, 30000], [10000, 20000, 30000], [11000, 21000], [12000, 22000]],
                [[5000, 5200, 5100], [4000, 4200, 4100], [4, 5], [40, 45]]):
            np.testing.assert_array_equal(renderer.data_source.data['timestamp'], times)
            np.testing.assert_array_equal(renderer.data_source.data['elapsed_us'], values)
        self.assertIsInstance(plot.y_scale, LogScale)
        self.assertIsInstance(plot.yaxis[0].ticker, LogTicker)
        self.assertEqual(plot.yaxis[0].ticker.mantissas, [1])
        self.assertEqual(plot.yaxis[0].ticker.num_minor_ticks, 0)
        self.assertIsInstance(plot.yaxis[0].formatter, LogTickFormatter)
        self.assertEqual(plot.yaxis[0].formatter.min_exponent, 0)
        self.assertEqual((plot.y_range.start, plot.y_range.end), (1, 10000))

    def test_historical_names_and_single_sample_are_supported(self):
        data = [topic('om_mpc_command', [10000], solve_us=[5000], solver_us=[4000]),
                topic('control_allocator_status', [20000], instance=1,
                      allocation_running_time=[40])]
        plot = get_execution_time_plot(data, plot_config).bokeh_plot
        self.assertEqual(len(plot.renderers), 3)
        np.testing.assert_array_equal(plot.renderers[-1].data_source.data['elapsed_us'], [40])

    def test_unavailable_times_are_gaps_without_mutating_the_log(self):
        values = np.array([5000, 0, -1, np.nan, np.inf, 5500])
        data = [topic('om_mpc_command', np.arange(6) * 10000, mpc_compute_us=values)]
        source = get_execution_time_plot(data, plot_config).bokeh_plot.renderers[0].data_source.data
        np.testing.assert_array_equal(source['elapsed_us'], [5000, np.nan, np.nan, np.nan, np.nan, 5500])
        np.testing.assert_array_equal(data[0].data['mpc_compute_us'], values)

    def test_logs_without_positive_timing_data_have_no_empty_plot(self):
        for data in [[], [topic('control_allocator_status', [100], allocation_method=[7])],
                     [topic('om_mpc_command', [100], mpc_compute_us=[0])]]:
            self.assertIsNone(get_execution_time_plot(data, plot_config))

    @unittest.skipUnless(os.environ.get('FLIGHT_REVIEW_TEST_ULOG'), 'Set a real ULog path')
    def test_real_log_loader_retains_mpc_and_allocation_times(self):
        ulog = load_ulog_file(os.environ['FLIGHT_REVIEW_TEST_ULOG'])
        ulog.get_dataset('om_mpc_command')
        plot = get_execution_time_plot(ulog.data_list, plot_config).bokeh_plot
        self.assertEqual(len(plot.renderers), 4)
        self.assertGreater(plot.y_range.start, 0)


if __name__ == '__main__':
    unittest.main()
