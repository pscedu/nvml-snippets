from pynvml import *


def return_gpu_indices(device_count=0, print_output=False) -> tuple:
    indices = tuple([i for i in range(device_count)])
    if print_output:
        print("export CUDA_VISIBLE_DEVICES=", end="")
        print(*indices, sep=",")
    return indices


def return_gpu_uuids(device_count=0, print_output=False) -> tuple:
    uuids = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(index=i)
        uuid = nvmlDeviceGetUUID(handle=handle)
        uuids.append(uuid.decode("utf-8"))
    if print_output:
        print("export CUDA_VISIBLE_DEVICES=", end="")
        print(*uuids, sep=",")
    return tuple(uuids)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    return_gpu_indices(device_count=deviceCount, print_output=True)
    return_gpu_uuids(device_count=deviceCount, print_output=True)
    nvmlShutdown()
