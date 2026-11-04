"""Plot recorded MPC and per-matrix allocation call times on one log axis."""

from types import SimpleNamespace

import numpy as np
from bokeh.models import LogScale, LogTicker, LogTickFormatter, Range1d

from config import colors8
from plotting import DataPlot


def get_execution_time_plot(data, plot_config, changed_params=None, x_range=None):
    """Use native timestamps and microseconds, accepting historical ULog fields.

    Zero/nonfinite times are unavailable on a log axis, not zero-cost calls.
    Prefer per-call allocator diagnostics over the historical status snapshot.
    """
    datasets = {(topic.name, topic.multi_id): topic for topic in data}
    series = []
    positive_times = []

    def add_series(topic, fields, name, legend, color):
        if topic is None:
            return
        field = next((key for key in fields if key in topic.data), None)
        if field is None:
            return
        timestamps = np.asarray(topic.data.get('timestamp', []))
        values = np.asarray(topic.data[field], dtype=float)
        valid = np.isfinite(values) & (values > 0)
        if len(timestamps) != len(values) or not np.any(valid):
            return
        positive_times.append(values[valid])
        series.append((SimpleNamespace(
            name=name, multi_id=0,
            data={'timestamp': timestamps, 'elapsed_us': np.where(valid, values, np.nan)}),
            legend, color))

    mpc = datasets.get(('om_mpc_command', 0))
    add_series(mpc, ('mpc_compute_us', 'solve_us'), 'mpc_compute',
               'MPC complete computation', colors8[0])
    add_series(mpc, ('qp_solve_us', 'solver_us'), 'qp_solve',
               'MPC QP solve/check section', colors8[1])

    instances = sorted({instance for name, instance in datasets
                        if name in ('control_allocator_diagnostics', 'control_allocator_status')})
    for instance in instances:
        allocation = datasets.get(('control_allocator_diagnostics', instance))
        if allocation is None or 'allocation_running_time' not in allocation.data:
            allocation = datasets.get(('control_allocator_status', instance))
        add_series(allocation, ('allocation_running_time',), f'allocation_{instance}',
                   f'Allocation {instance}: complete allocate()', colors8[2 + instance % 6])

    if not series:
        return None
    lower = int(np.floor(np.log10(min(np.min(values) for values in positive_times))))
    upper = max(lower + 1,
                int(np.ceil(np.log10(max(np.max(values) for values in positive_times)))))
    plot_data = [topic for topic, _, _ in series]
    data_plot = DataPlot(
        plot_data, plot_config, plot_data[0].name,
        title='MPC & Control Allocation Execution Time', y_axis_label='Elapsed time [us]',
        y_range=Range1d(10.0 ** lower, 10.0 ** upper),
        changed_params=changed_params, x_range=x_range)
    for topic, legend, color in series:
        data_plot.change_dataset(topic.name)
        data_plot.add_graph(['elapsed_us'], [color], [legend],
                            use_downsample=False, break_on_timestamp_gaps=True)
    if data_plot.finalize() is None:
        return None

    plot = data_plot.bokeh_plot
    plot.name = 'control_execution_time'
    plot.y_scale = LogScale()
    ticker = LogTicker(mantissas=[1], num_minor_ticks=0)
    plot.yaxis[0].ticker = ticker
    plot.yaxis[0].formatter = LogTickFormatter(ticker=ticker, min_exponent=0)
    plot.yaxis[0].minor_tick_line_color = None
    plot.ygrid.ticker = ticker
    plot.ygrid.minor_grid_line_color = None
    return data_plot
