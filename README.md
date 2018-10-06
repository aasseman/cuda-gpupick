# cuda-gpupick
Automated unused CUDA device and NUMA node binding for any command.
It will give you the GPUs that:
- are as far as possible in the topology from those that are already in use.
- AND, if you need more than one, the GPUs that are the closest to each other.

Tested only on Ubuntu 16.04. Should work on any linux distro.

# Usage

If you need to make 1 GPU available to your program:
```
$ cuda-gpupick -n1 [command]
```
Any value, including 0, is valid, as long as there are enough GPUs available to fulfill your request.

# Install

Clone this repository, and then:
```
$ cd cuda-gpupick/
$ sudo -H pip3 install .
```
That will install the `cuda-gpupick` command system-wide.
