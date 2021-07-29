from linprog import simplex, UnboundedLinearProgrammingError, UnsolvableLinearProgrammingError
import numpy as np

def read_array():
  str = input()
  return [int(x) for x in str.split()]

def arr_to_s(arr):
  return ' '.join([str(format(x, ".7f")) for x in arr])

def main():
  m, n = read_array()
  A = np.ndarray((m, n))
  b = np.zeros(m)
  
  c = np.array(read_array())

  for i in range(m):
    lin = read_array()
    A[i, :] = lin[:-1]
    b[i] = lin[-1]

  try:
    x, value, certificate = simplex(c, A, b)
  except AssertionError as e:
    print("Invalid linear programming: ", str(e))
  except UnboundedLinearProgrammingError as e:
    print('ilimitada')
    print(arr_to_s(e.solution))
    print(arr_to_s(e.certificate))
  except UnsolvableLinearProgrammingError as e:
    print('inviavel')
    print(arr_to_s(e.certificate))
  else:
    print('otima')
    print(value)
    print(arr_to_s(x))
    print(arr_to_s(certificate))

if __name__=="__main__":
  main()