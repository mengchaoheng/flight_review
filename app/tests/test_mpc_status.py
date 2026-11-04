"""Regression tests for MPC status plots and optional ULog integration.

Run with PYTHONPATH=app/plot_app python -m unittest discover -s app/tests -v.
Set FLIGHT_REVIEW_TEST_ULOG to also exercise the real log loader.
"""

import os
from types import SimpleNamespace
import unittest

import numpy as np
from bokeh.models import Range1d, Step

from config import plot_config
from helper import load_ulog_file
from mpc_status import get_mpc_status_plot


def topic(name, timestamps, **fields):
    """Construct the subset of a ULog dataset required by DataPlot."""
    return SimpleNamespace(
        name=name, multi_id=0,
        data={'timestamp': np.asarray(timestamps, dtype=np.uint64),
              **{key: np.asarray(value, dtype=np.uint8)
                 for key, value in fields.items()}})


class MpcStatusTest(unittest.TestCase):
    """Check flag provenance, missing topics, gaps and stale final samples."""

    def test_distinct_lanes_use_final_arbitration(self):
        """Applied flags must come from arbitration, not bridge computation."""
        data = [
            topic('om_mpc_control_status', [100, 200, 300], mpc_active=[0, 1, 1],
                  translational_indi_active=[0, 0, 1]),
            topic('om_mpc_acc_indi_command', [110, 210, 310], valid=[1, 1, 1]),
            topic('om_mpc_disturbance', [110, 210, 310], valid=[0, 0, 1]),
        ]
        result = get_mpc_status_plot(data, plot_config)
        plot = result.bokeh_plot
        self.assertEqual(plot.title.text, 'MPC Control Status')
        self.assertEqual(len(plot.renderers), 3)
        self.assertTrue(all(isinstance(renderer.glyph, Step)
                            for renderer in plot.renderers))
        expected = [('mpc_active', [4, 5, 5]),
                    ('translational_indi_active', [2, 2, 3]),
                    ('valid', [0, 0, 1])]
        for renderer, (field, values) in zip(plot.renderers, expected):
            np.testing.assert_array_equal(renderer.data_source.data[field], values)
        self.assertEqual(plot.yaxis[0].major_label_overrides[5], 'MPC: 1')

    def test_old_log_has_no_empty_plot(self):
        """Older logs retain their existing INDI plot without an MPC plot."""
        data = [topic('rate_ctrl_status', [100, 200], indi_active=[0, 1])]
        self.assertIsNone(get_mpc_status_plot(data, plot_config))
        self.assertIsNone(get_mpc_status_plot(
            [topic('om_mpc_control_status', [100, 200], unsupported=[0, 1])], plot_config))

    def test_partial_log_draws_only_supported_fields(self):
        """Absent flags must not be fabricated as inactive."""
        data = [topic('om_mpc_control_status', [100, 200], mpc_active=[0, 1])]
        plot = get_mpc_status_plot(data, plot_config).bokeh_plot
        self.assertEqual(len(plot.renderers), 1)
        np.testing.assert_array_equal(
            plot.renderers[0].data_source.data['mpc_active'], [0, 1])

    def test_gaps_break_lines_without_modifying_log(self):
        """Dropouts stay unknown and bridge flags end at their last sample."""
        timestamps = [100000, 200000, 300000, 1000000, 1100000]
        data = [topic('om_mpc_disturbance', timestamps,
                      valid=[0, 0, 0, 1, 1])]
        plot = get_mpc_status_plot(
            data, plot_config, x_range=Range1d(0, 2000000)).bokeh_plot
        source = plot.renderers[0].data_source.data
        self.assertEqual(np.count_nonzero(np.isnan(source['valid'])), 1)
        self.assertEqual(source['timestamp'][-1], 1100000)
        self.assertEqual(source['valid'][-1], 1)
        first_after_gap = np.flatnonzero(source['timestamp'] == 1000000)[0]
        self.assertEqual(source['valid'][first_after_gap], 1)
        np.testing.assert_array_equal(data[0].data['valid'], [0, 0, 0, 1, 1])

    def test_normal_publish_rate_changes_are_not_gaps(self):
        """Both slower and faster bridge updates remain visible."""
        timestamps = np.cumsum([0] + [20000] * 20 + [10000] * 80)
        data = [topic('om_mpc_disturbance', timestamps,
                      valid=[0] * 101)]
        plot = get_mpc_status_plot(data, plot_config).bokeh_plot
        source = plot.renderers[0].data_source.data
        self.assertFalse(np.any(np.isnan(source['valid'])))
        np.testing.assert_array_equal(source['timestamp'], timestamps[[0, -1]])

    def test_long_constant_trace_preserves_gap_and_transition(self):
        """log_137's long constant traces must not crash BokehJS indexing."""
        timestamps = np.arange(15000, dtype=np.uint64) * 20000
        timestamps[9000:] += 30000000
        values = np.zeros(15000, dtype=np.uint8)
        values[14000:] = 1
        data = [topic('om_mpc_disturbance', timestamps,
                      valid=values)]
        source = get_mpc_status_plot(
            data, plot_config).bokeh_plot.renderers[0].data_source.data
        self.assertLess(len(source['timestamp']), 16)
        self.assertEqual(source['timestamp'][0], timestamps[0])
        self.assertEqual(source['timestamp'][-1], timestamps[-1])
        self.assertEqual(np.count_nonzero(np.isnan(source['valid'])), 1)
        gap_index = np.flatnonzero(np.isnan(source['valid']))[0]
        self.assertEqual(source['timestamp'][gap_index - 1], timestamps[8999])
        self.assertEqual(source['timestamp'][gap_index + 1], timestamps[9000])
        switch_index = np.flatnonzero(source['valid'] == 1)[0]
        self.assertEqual(source['timestamp'][switch_index], timestamps[14000])
        np.testing.assert_array_equal(data[0].data['valid'], values)

    def test_historical_log_uses_final_status_and_old_disturbance(self):
        data = [
            topic('om_mpc_status', [100, 200, 300], active=[0, 1, 1],
                  acceleration_indi_active=[0, 0, 1]),
            topic('om_mpc_indi_status', [100, 200, 300],
                  acceleration_active=[1, 1, 1], disturbance_valid=[0, 0, 1]),
        ]
        plot = get_mpc_status_plot(data, plot_config).bokeh_plot
        expected = [('active', [4, 5, 5]),
                    ('acceleration_indi_active', [2, 2, 3]),
                    ('disturbance_valid', [0, 0, 1])]
        for renderer, (field, values) in zip(plot.renderers, expected):
            np.testing.assert_array_equal(renderer.data_source.data[field], values)

    def test_new_status_wins_over_conflicting_historical_topic(self):
        data = [
            topic('om_mpc_control_status', [100, 200], mpc_active=[0, 1],
                  translational_indi_active=[0, 0]),
            topic('om_mpc_status', [100, 200], active=[1, 1],
                  acceleration_indi_active=[1, 1]),
            topic('om_mpc_acc_indi_command', [100, 200], valid=[1, 1]),
            topic('om_mpc_indi_diagnostics', [100, 200], corrected_rates=[1, 1]),
        ]
        plot = get_mpc_status_plot(data, plot_config).bokeh_plot
        self.assertEqual(len(plot.renderers), 2)
        np.testing.assert_array_equal(plot.renderers[0].data_source.data['mpc_active'], [2, 3])
        np.testing.assert_array_equal(plot.renderers[1].data_source.data['translational_indi_active'], [0, 0])

    def test_diagnostics_and_command_validity_are_not_application(self):
        data = [topic('om_mpc_acc_indi_command', [100, 200], valid=[1, 1]),
                topic('om_mpc_indi_diagnostics', [100, 200], corrected_rates=[1, 1])]
        self.assertIsNone(get_mpc_status_plot(data, plot_config))

    @unittest.skipUnless(os.environ.get('FLIGHT_REVIEW_TEST_ULOG'),
                         'Set FLIGHT_REVIEW_TEST_ULOG for real log coverage')
    def test_real_log_loader_and_plot(self):
        """The same loader used by serve.py must retain both MPC topics."""
        ulog = load_ulog_file(os.environ['FLIGHT_REVIEW_TEST_ULOG'])
        names = {topic.name for topic in ulog.data_list}
        self.assertTrue('om_mpc_control_status' in names or 'om_mpc_status' in names)
        self.assertTrue('om_mpc_disturbance' in names or 'om_mpc_indi_status' in names)
        result = get_mpc_status_plot(ulog.data_list, plot_config)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.bokeh_plot.renderers), 3)


if __name__ == '__main__':
    unittest.main()
