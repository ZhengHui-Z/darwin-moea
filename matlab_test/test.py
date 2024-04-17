import matlab
import numpy as np
import matlab.engine

eng = matlab.engine.start_matlab()

p = np.random.random((100, 3))
p = matlab.double(p.tolist())
ref = eng.GA(p)
