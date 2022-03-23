import qs

stream_id = 'Blodgett'
reach_slope = 0.02
discharge_interval = 900
# minimum_fraction = 0  # the default value is 0.01 but it can be changed and added as a param below.

qs.FractionalTransport(stream_id, reach_slope, discharge_interval)
