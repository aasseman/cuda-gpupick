#!/usr/bin/python3

# Copyright (c) 2018 Alexis Asseman
# Licensed under MIT (https://github.com/aasseman/cuda-gpupick/blob/master/LICENSE)

from py3nvml.py3nvml import *
import os
import argparse
import math
import numpy as np
from sys import stderr

# Colors for terminal text
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=0, help='Number of requested GPUs (default: %(default)s)')
    parser.add_argument('-f', action='store_true', default=False, help='Ignore NUMA crossing (default: %(default)s)')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    requested_devices_count = args.n
    force_request = args.f
    print(force_request)

    if requested_devices_count > 0:
        nvmlInit()

        print("Nvidia driver:", nvmlSystemGetDriverVersion())
        print("CUDA device count:", nvmlDeviceGetCount())

        device_count = nvmlDeviceGetCount()

        if requested_devices_count > device_count:
            print(bcolors.FAIL + "FATAL ERROR: " + str(requested_devices_count) + " CUDA devices requested, while only "
                  + str(device_count) + " are visible on this system." + bcolors.ENDC, file=stderr)
            exit(-1)

        # Set a binary code representing the availability of each GPU
        # The "further" a GPU is from a used one, the more available it is
        # Start the algo with 0000000 -> all GPUs are fully available
        device_available = [0b0000000 for i in range(device_count)]
        #                     ^     ^
        #                     |     └> Most available
        #                     └> Least available (Not available at all)

        device_available_count = device_count

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            print("Device", i, ":", nvmlDeviceGetName(handle))
            mem_usage = nvmlDeviceGetMemoryInfo(handle).used / nvmlDeviceGetMemoryInfo(handle).total
            compute_processes_count = len(nvmlDeviceGetComputeRunningProcesses(handle))
            compute_utilization = nvmlDeviceGetUtilizationRates(handle).gpu
            print("\tNumber of compute processes: " + str(compute_processes_count))
            print("\tMemory utilization: " + str(math.ceil(100 * mem_usage)) + '%')
            print("\tCompute utilization: " + str(compute_utilization) + '%')

            if compute_processes_count > 0:
                device_available_count = device_available_count - 1

                for j in range(device_count):
                    if i == j:
                        device_available[j] = device_available[j] | (1 << 6)
                    else:
                        level = nvmlDeviceGetTopologyCommonAncestor(nvmlDeviceGetHandleByIndex(i),
                                                                          nvmlDeviceGetHandleByIndex(j))
                        device_available[j] = device_available[j] | (1 << (6 - (level // 10)))

        if device_available_count < requested_devices_count:
            print(bcolors.FAIL + "FATAL ERROR: " + str(requested_devices_count) + " devices requested, but only "
                  + str(device_available_count) + " are available." + bcolors.ENDC, file=stderr)
            exit(-1)

        # Sort the devices in decreasing availability order (increasing binary scores)
        device_available_argsort = np.argsort(device_available)
        if device_available_argsort[0] == (1 << 6):
            print(bcolors.FAIL + "FATAL ERROR: No available devices." + bcolors.ENDC, file=stderr)
            exit(-1)

        chosen_gpus_indices = []
        # Try to choose the requested number of GPUs that are as close as possible to the most available one.
        # If not possible, try with the next most available, etc, until running out of free GPUs (-> abort)
        device_index = device_available_argsort[0]

        if device_available[device_index] == (1 << 6):
            print(bcolors.FAIL + "FATAL ERROR: Can't choose " + str(requested_devices_count)
                  + " GPUs without crossing NUMA barrier." + bcolors.ENDC, file=stderr)
            exit(-1)

        chosen_gpus_indices = [device_index]
        level = 10 # Interconnect level as defined in Nvidia's NVML library

        def NearestGpus(level):
            near_gpus = []
            for d in device_available_argsort:
                if d != device_index:
                    _level = nvmlDeviceGetTopologyCommonAncestor(nvmlDeviceGetHandleByIndex(device_index),
                                                                 nvmlDeviceGetHandleByIndex(d))
                    if _level == level:
                        near_gpus += [nvmlDeviceGetHandleByIndex(d)]
            return near_gpus

        max_level = NVML_TOPOLOGY_SYSTEM if force_request else NVML_TOPOLOGY_HOSTBRIDGE

        while len(chosen_gpus_indices) < requested_devices_count and level <= max_level:
            for h in NearestGpus(level):
                i = nvmlDeviceGetIndex(h)
                if i not in chosen_gpus_indices and not device_available[i] & (1 << 6):
                    chosen_gpus_indices.append(i)
            level = level + 10

        if len(chosen_gpus_indices) < requested_devices_count:
            print(bcolors.FAIL + "FATAL ERROR: Can't choose " + str(requested_devices_count)
                  + " GPUs without crossing NUMA barrier." + bcolors.ENDC, file=stderr)
            exit(-1)
        elif len(chosen_gpus_indices) >= requested_devices_count:
            chosen_gpus_indices = chosen_gpus_indices[0:requested_devices_count]

        print('\n')
        print("Chosen GPU(s): ", chosen_gpus_indices)

        chosen_gpus_handles = [nvmlDeviceGetHandleByIndex(i) for i in chosen_gpus_indices]

        pci_list = [nvmlDeviceGetPciInfo(h).busId.decode('utf-8') for h in chosen_gpus_handles]

        numa_nodes = []
        for i in pci_list:
            with open("/sys/class/pci_bus/" + i[0:7] + "/device/numa_node", 'r') as f:
                numa_nodes.append(int(f.read()))

        numa_nodes = np.unique(numa_nodes)

        if len(numa_nodes) > 1 and not force_request:
            print(bcolors.FAIL + "INTERNAL ERROR: Picked GPUs from more than 1 NUMA node: " + str(numa_nodes) + bcolors.ENDC, file=stderr)
            exit(-1)

        numa_node = numa_nodes[0]
        print("NUMA node: " + str(numa_node))

        nvmlShutdown()

    elif requested_devices_count == 0:
        chosen_gpus_indices = []

    command = ' '.join(args.command)
    command_cuda = "CUDA_VISIBLE_DEVICES=" + ','.join([str(i) for i in chosen_gpus_indices])
    if requested_devices_count > 0:
        command_numactl = "numactl -N " + str(numa_node) + " --preferred " + str(numa_node)
    else:
        command_numactl = ""
    print(command_cuda)
    print(command_numactl)
    command_appended = ' '.join([command_cuda, command_numactl, command])
    print(command_appended)

    os.system(command_appended)


if __name__ == '__main__':
    main()
