# dynamic-stress-partitioning
Surface-based fractional sediment transport formula based on dynamic shear stress partitioning framework.

## Setup
Download the repository. Navigate to the `scripts` folder, open a shell terminal, and run the bootstrap.sh script. This 
will set up a virtual environment with all of the necessary Python packages.

## Data Entry
Navigate to the `Input_data` folder, where three csv files are located. 
- Enter grain size measurements (in mm) from the reach of interest into the 'grain_size.csv' file. 
- Enter hydraulic measurements into the 'hydraulic_geometry.csv' file. You should have three more paired measurements
of discharge (cms), channel width (m), and average depth (m).
- Enter the discharge (cms) record for which you want to calculate sediment transport in the 'discharge.csv' file. 

## Running the model
The simplest way to run the model is via command line. Open a command line terminal in the model folder and activate the 
python environment. (These instructions apply to UNIX systems, Windows still to come).

```commandline
source .venv/bin/activate
```
Then change directories into the `dsp_transport` folder, where the python scripts are located.

```commandline
cd dsp_transport
```

and run the model using the usage below. 'dsp_transport.py' calculates fractional transport, storing results in a .csv 
file in the `Outputs` folder. 'plotting.py' produces some plots of the outputs and stores them in the same output 
folder.

```
usage: dsp_transport.py [-h] [--minimum_fraction MINIMUM_FRACTION]
                        stream_id reach_slope discharge_interval

positional arguments:
  stream_id             The name of the stream as entered in the Input_data
                        csv files
  reach_slope           A value for the reach averaged slope
  discharge_interval    The time (in seconds) between each discharge
                        measurement in the discharge csv file

optional arguments:
  -h, --help            show this help message and exit
  --minimum_fraction MINIMUM_FRACTION
                        (optional) A minimum fraction to assign to fine grain
                        size classes

```

```
usage: plotting.py [-h] stream_id

positional arguments:
  stream_id   The name of the stream as entered in the Input_data csv files

optional arguments:
  -h, --help  show this help message and exit

```

for example, if I entered data for 'Beaver_Creek' in the input tables, it's slope is 0.015 and the interval for 
discharge measurements is 15 minutes, I would enter:

```commandline
python dsp_transport.py Beaver_Creek 0.015 900
```