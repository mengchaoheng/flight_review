"""Status steps retain exact switches and dropout boundaries when compacted."""
import unittest
from types import SimpleNamespace

import numpy as np
from bokeh.models import Range1d

from config import plot_config
from plotting import DataPlot, compact_step_data


class StepDataTest(unittest.TestCase):
    def test_multiple_fields_preserve_every_transition(self):
        data = {'timestamp': np.arange(12),
                'a': np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                'b': np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])}
        result = compact_step_data(data)
        np.testing.assert_array_equal(result['timestamp'], [0, 2, 3, 6, 7, 11])
        # The held value at each original sample must remain identical.
        indexes = np.searchsorted(result['timestamp'], data['timestamp'], side='right') - 1
        for field in ['a', 'b']:
            np.testing.assert_array_equal(result[field][indexes], data[field])

    def test_indi_status_preserves_real_samples_around_dropout(self):
        timestamps = np.arange(3000, dtype=np.uint64) * 100000
        timestamps[2000:] += 1000000
        values = np.zeros(3000, dtype=np.uint8)
        values[100:] = 1
        topic = SimpleNamespace(name='rate_ctrl_status', multi_id=0,
                                data={'timestamp': timestamps, 'indi_active': values})
        plot = DataPlot([topic], plot_config, topic.name, x_range=Range1d(0, 400000000))
        plot.add_graph(['indi_active'], ['blue'], ['Rate INDI'],
                       use_downsample=False, use_step_lines=True,
                       break_on_timestamp_gaps=True)
        result = plot.bokeh_plot.renderers[0].data_source.data
        self.assertLess(len(result['timestamp']), 16)
        gap = np.flatnonzero(np.isnan(result['indi_active']))[0]
        self.assertEqual(result['timestamp'][gap-1], timestamps[1999])
        self.assertEqual(result['timestamp'][gap+1], timestamps[2001])
        switch = np.flatnonzero(result['indi_active'] == 1)[0]
        self.assertEqual(result['timestamp'][switch], timestamps[100])
        self.assertEqual(result['timestamp'][-1], timestamps[-1])
        self.assertEqual(len(topic.data['timestamp']), 3000)
        np.testing.assert_array_equal(topic.data['indi_active'], values)


if __name__ == '__main__':
    unittest.main()
