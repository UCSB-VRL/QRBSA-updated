import numpy as np

def custom_float(x):
    return float(x.replace('−', '-'))

try:
    data = np.loadtxt('quaternions_edge_fz.txt', converters={i: custom_float for i in range(10)})
    np.savetxt('quaternions_edge_fz_fixed.txt', data, fmt='%.10f')
    print("Fixed file saved as quaternions_edge_fz_fixed.txt")
except Exception as e:
    print(f"Error: {e}")
