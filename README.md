# dynamic-stress-partitioning
Surface-based fractional sediment transport formula based on dynamic shear stress partitioning framework.

## Setup
Download the .zip of the repository from the latest release. Navigate to the `scripts` folder, open a shell 
terminal, and run the bootstrap.sh script. This will set up a virtual environment with all of the necessary 
Python packages.

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
folder. 'generate_gsd.py' either uses existing pebble count data to generate a grain size distribution, or generates
a new distribution by providing D16, D50, and D84 values.

```
usage: dsp_transport.py [-h] [--minimum_fraction MINIMUM_FRACTION]
                        [--lwd_factor LWD_FACTOR]
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
  --lwd_factor LWD_FACTOR
                        Alters the critical shields stress to account for the
                        affects of large wood, None is no wood present, 1 is
                        some scattered pieces, 2 is wood throughout the reach
                        and 3 is jams present

```

```
usage: plotting.py [-h] stream_id

positional arguments:
  stream_id   The name of the stream as entered in the Input_data csv files

optional arguments:
  -h, --help  show this help message and exit

```

```
usage: generate_gsd.py [-h] [--csv_in CSV_IN] [--d50 D50] [--d16 D16]
                       [--d84 D84]
                       csv_out

positional arguments:
  csv_out          Path to save the output csv of generated grain size data

optional arguments:
  -h, --help       show this help message and exit
  --csv_in CSV_IN  Path to a csv file containing grain size data; used to
                   estimate distribution parameters and find D50, D16, and D84
                   if not provided.If no input csv is entered, values MUST be
                   provided for D50, D16, and D84The csv should have a column
                   with header "D" containing grain size countsin millimeters
  --d50 D50        A value for D50 (mm). If no input csv is entered, a value
                   for D50 must be provided. If an input csv IS entered, the
                   calculated D50 can be overridden by providing a value here
  --d16 D16        A value for D16 (mm). If no input csv is entered, a value
                   for D16 must be provided. If an input csv IS entered, the
                   calculated D16 can be overridden by providing a value here
  --d84 D84        A value for D84 (mm). If no input csv is entered, a value
                   for D84 must be provided. If an input csv IS entered, the
                   calculated D84 can be overridden by providing a value here
```

for example, if I entered data for 'Beaver_Creek' in the input tables, it's slope is 0.015 and the interval for 
discharge measurements is 15 minutes, I would enter:

```
python dsp_transport.py 'Beaver_Creek' 0.015 900
```

In addition to the self-contained program, the `calculate_transport.py` script is designed to be imported by other 
programs in order to use the transport function in other applications. 

```
usage: calculate_transport.py [-h] fractions slope discharge depth width interval

positional arguments:
  fractions   A python dictionary {size: fraction in bed} with all size classes and thefraction of the bed they make up e.g. {0.004: 0.023, 0.008: 0.041}
  slope       The reach slope
  discharge   The flow discharge in m3/s
  depth       The average flow depth at the given discharge
  width       The average flow width at the given discharge
  interval    The length of time in seconds of the discharge measurement

optional arguments:
  -h, --help  show this help message and exit
```
