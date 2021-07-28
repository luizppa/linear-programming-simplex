from linprog import simplex
import numpy as np
import sys

def main():
  c = np.array([3, 2, 4])
  A = np.array([[1, 1, 2],
                [2, 0, 3],
                [2, 1, 3]])
  b = np.array([4, 5, 7])

  try:
    x, value, certificate = simplex(c, A, b)
  except AssertionError as e:
    print("Invalid linear programming: ", str(e))
  except (UnboundedLinearProgrammingError, UnsolvableLinearProgrammingError) as e:
    print(str(e))
  else:
    print("Solution: ", str(x))
    print("Optimal value: ", str(value))
    print("Certificate: ", str(certificate))

if __name__=="__main__":
  main()