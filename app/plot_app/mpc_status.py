"""Plot actual MPC ownership and middle-layer INDI feedback states."""

import numpy as np
from bokeh.models import FixedTicker, Range1d

from config import colors8
from plotting import DataPlot


def _break_status_gaps(renderer, field):
    """Keep valid samples and insert gaps, allowing normal publish-rate changes."""
    source = renderer.data_source
    timestamps = np.asarray(source.data['timestamp'], dtype=np.int64)
    intervals = np.diff(timestamps)
    positive_intervals = intervals[intervals > 0]
    if not len(positive_intervals):
        return

    # The bridge changes rate when acc INDI becomes active. The median alone
    # incorrectly treats its slower, valid publication interval as a dropout.
    # Select an observed interval so interpolation toward a large dropout
    # cannot inflate the threshold in short datasets.
    normal_interval = np.sort(positive_intervals)[
        int(0.9 * (len(positive_intervals) - 1))]
    gap_indexes = np.flatnonzero(intervals > normal_interval * 1.5) + 1
    if not len(gap_indexes):
        return

    source.data = {
        'timestamp': np.insert(timestamps, gap_indexes, timestamps[gap_indexes - 1] + 1),
        field: np.insert(np.asarray(source.data[field], dtype=float), gap_indexes, np.nan),
    }


def get_mpc_status_plot(data, plot_config, changed_params=None, x_range=None):
    """Return a status plot, or None when no supported flags were logged.

    Use final arbitration for MPC/INDI application and the bridge's validity
    flag for disturbance feedback. Separate 0/1 lanes keep overlapping active
    intervals visible. Never extend a stopped topic to the end of the flight.
    """
    datasets = {topic.name: topic for topic in data if topic.multi_id == 0}
    fields = [
        ('om_mpc_status', 'active', 'MPC applied', 'MPC', colors8[2]),
        ('om_mpc_status', 'acceleration_indi_active',
         'Middle-layer acc INDI applied', 'acc INDI', colors8[0]),
        ('om_mpc_indi_status', 'disturbance_valid',
         'Disturbance feedback valid', 'Feedback', colors8[1]),
    ]
    available = [
        spec for spec in fields
        if spec[0] in datasets and spec[1] in datasets[spec[0]].data
        and len(datasets[spec[0]].data.get('timestamp', [])) > 0
    ]
    if not available:
        return None

    data_plot = DataPlot(
        data, plot_config, available[0][0], title='MPC Status',
        y_axis_label='Status (separate 0/1 lanes)',
        y_range=Range1d(-0.25, 2 * len(available) - 0.75),
        changed_params=changed_params, x_range=x_range)
    tick_labels = {}
    for index, (topic, field, legend, label, color) in enumerate(available):
        offset = 2 * (len(available) - index - 1)
        data_plot.change_dataset(topic)
        data_plot.add_graph(
            [lambda values, key=field, lane=offset:
             (key, np.asarray(values[key], dtype=float) + lane)],
            [color], [legend], use_downsample=False, use_step_lines=True)
        _break_status_gaps(data_plot.bokeh_plot.renderers[-1], field)
        tick_labels[offset] = label + ': 0'
        tick_labels[offset + 1] = label + ': 1'

    if data_plot.finalize() is None:
        return None
    plot = data_plot.bokeh_plot
    plot.name = 'mpc_status'
    legend = plot.legend[0]
    plot.center = [item for item in plot.center if item is not legend]
    legend.orientation = 'horizontal'
    plot.add_layout(legend, 'above')
    plot.yaxis[0].ticker = FixedTicker(ticks=sorted(tick_labels))
    plot.yaxis[0].major_label_overrides = tick_labels
    return data_plot
