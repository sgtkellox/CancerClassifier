from multiprocessing import Process
import os

from preprocessing_slides import process_tiles
from preprocessing_slides import mask4
from preprocessing_slides import SIZE



if __name__ == "__main__":
    processes = []
    num_processes = 16
    slidePath = r""

    for slide in os.listdir(slidePath):
        print()

    # create processes and asign a function for each process
    for i in range(num_processes):
        process = Process(target=process_tiles)
        processes.append(process)

    # start all processes
    for process in processes:
        process.start()

    # wait for all processes to finish
    # block the main thread until these processes are finished
    for process in processes:
        process.join()