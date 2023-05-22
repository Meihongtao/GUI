import multiprocessing
from multiprocessing import Pool



if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    data = pd.read_csv("data\\2023-3-30-点编号.csv")
    print(data)
    d = np.asarray(data.values[:,1:])
    print(d)
    print(d.shape)
    a = d.reshape(1,-1)
    print(a)
    np.savetxt('363-points.txt', d, delimiter=',', newline=',', fmt='%d')
    # pool = multiprocessing.Pool()
    # total_count = 10
    # callback.counter = 0

    # for i in range(total_count):
    #     pool.apply_async(my_function, (i,), callback=callback)

    # pool.close()
    # pool.join()
