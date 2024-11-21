# RCLoc - Practical Root Cause Localization for FPGA Simulation Tools via Fault Multiple Dimensions Analysis

## Introduction
RCLoc, a fault root cause localization method for FPGA simulation tools.\
RCLoc involves three stages: pre-training to compute the confidence matrix, redundancy removal of test cases, and training to predict the fault root cause location.

## Datasets
We used an fault root cause localization dataset constructed for FPGA simulation tools Iverilog and Verilator, which includes two levels of granularity: file-level and statement-level. 
The two FPGA simulation tools contain 45 and 96 real bugs, respectively, that have been fixed in practical applications.


| Type of Information        | Iverilog              | Verilator             |
|----------------------------|-----------------------|-----------------------|
| Number of Bugs             | 45                    | 96                    |
| Number of Features         | 175                   | 175                   |
| Granularity Level          | File, Statement       | File, Statement       |
| File Level Train Size      | 6,000+ for every bug  | 4,000+ for every bug  |
| Statement Level Train Size | 25,000+ for every bug | 67,000+ for every bug |

## Requirements:
- Python 3.7.12
- Tensorflow-gpu 1.15.0
- Cleanlab 1.0.1
- protobuf 3.20.1
- pandas 1.3.5

## Replication

#### Pretrain && Clean && Retrain

Use ```python ./Iverilog/main.py``` to start fault root cause localization for Iverilog, the optional CL threshold parameter, as described in ```main.py```, has a default value of 0.0007. 

Use ```python ./Verilator/main.py``` to start fault root cause localization for Verilator, the optional CL threshold parameter, as described in ```main.py```, has a default value of 0.0001. 

The results are stored in ```./Iverilog/Result/``` or ```./Verilator/Result/```.

#### Evaluate RCLoc

The corresponding evaluation functions have already been called in the ```main.py```. If you wish to use them separately, 
you can find ```rank_file.py``` and ```rank_statement.py``` in the ```./Iverilog/util/``` or ```./Verilator/util/```, which correspond to file-level and statement-level evaluation methods, respectively.
