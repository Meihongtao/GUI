import multiprocessing
from multiprocessing import Pool

def my_function(x):
    return x * x

def callback(result):
    completed_count = callback.counter + 1
    callback.counter = completed_count

    progress = (completed_count / total_count) * 100
    print("Progress: {:.2f}%".format(progress))

if __name__ == "__main__":
    import numpy as np
    a = np.load("D:\DESKTOP\GUI\data\outputs\\batch-1.npy")
    a = np.linspace(12,45,50)
    print(a.shape)
    # pool = multiprocessing.Pool()
    # total_count = 10
    # callback.counter = 0

    # for i in range(total_count):
    #     pool.apply_async(my_function, (i,), callback=callback)

    # pool.close()
    # pool.join()
