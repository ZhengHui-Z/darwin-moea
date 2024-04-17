import numpy as np
from darwin.ea.base.dominated import DominatedBase
from darwin.ea.operation.dominated import tools


class FastNonDominatedSort(DominatedBase):

    def do(self, obj_var):
        N = obj_var.shape[0]
        front = np.full(N, -1, dtype=np.int32)

        index = np.lexsort(np.rot90(obj_var))

        MaxFNo = 0
        for i in index:
            NowFNo = 0
            while True:
                Dominated = False
                for j in np.flipud(np.where(front == NowFNo)[0]):
                    if tools.dominate(obj_var[j], obj_var[i]):
                        Dominated = True
                        break
                if Dominated:
                    NowFNo += 1
                    if NowFNo > MaxFNo:
                        MaxFNo += 1
                else:
                    front[i] = NowFNo
                    break
        return front, MaxFNo


if __name__ == "__main__":
    fnds = FastNonDominatedSort()
    obj_var = np.random.random((100, 3))
    front, MaxFNo = fnds.do(obj_var)
