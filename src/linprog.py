# -*- coding: utf-8 -*-
import numpy as np
import math

class LinearProgrammingError(Exception):
  def __init__(self, *args):
    self.message = args[0]
    self.certificate = args[1]

  def __str__(self):
    return '{0}, certificate = {1}'.format(self.message, self.certificate)

class UnboundedLinearProgrammingError(LinearProgrammingError):
  def __init__(self, *args):
    super(UnboundedLinearProgrammingError, self).__init__('Optimal value is infinite', args[0])
    self.solution = args[1]

class UnsolvableLinearProgrammingError(LinearProgrammingError):
  def __init__(self, *args):
    super(UnsolvableLinearProgrammingError, self).__init__('Linear programming has no solution', args[0])

def is_optimal(tableaux):
  return all((c >= 0 or np.isclose(c, 0)) for c in tableaux[0, :-1])

def is_negative(arr):
  return all(a < 0 and not np.isclose(a, 0) for a in arr)

def is_canonical(arr):
  sum = 0

  for i in range(len(arr)):
    if not np.isclose(arr[i], 1) and not np.isclose(arr[i], 0):
      return False
    sum += arr[i]
  
  return np.isclose(sum, 1)

def build_tableaux(A, c, b):
  ncons, nvars = A.shape
  cn = [ -i for i in c ]

  tableaux = np.zeros((ncons + 1, nvars + ncons + 1))
  tableaux[1:, :nvars] = A
  tableaux[1:, nvars:-1] = np.identity(ncons)
  tableaux[0, :-1] = np.append(cn, np.zeros(ncons))
  tableaux[1:, -1] = b
  return tableaux

def choose_pivot(tableaux, col):
  values = np.zeros(tableaux.shape[0] - 1)
  
  for i in range(1, tableaux.shape[0]):
    if tableaux[i, col] > 0:
      values[i - 1] = tableaux[i, -1] / tableaux[i, col]
    else:
      values[i - 1] = math.inf
  
  return list(values).index(min(values)) + 1

def pivot_element(tableaux, operations, lin, col):
  pivot_value = tableaux[lin, col]
  tableaux[lin] = tableaux[lin] / pivot_value
  operations[lin] = operations[lin] / pivot_value
  
  for i in range(tableaux.shape[0]):
    if i != lin:
      multiplier = -tableaux[i, col]
      tableaux[i] = tableaux[i] + (tableaux[lin] * multiplier)
      operations[i] = operations[i] + (operations[lin] * multiplier)

def get_solution(tableaux):
  x = np.zeros(tableaux.shape[1] - 1)

  for i in range(len(x)):
    if is_canonical(tableaux[:, i]):
      x[i] = tableaux[list(tableaux[:, i]).index(1), -1]
      
  return x

def get_unbounded_certificate(tableaux, col):
  certificate = np.zeros(tableaux.shape[1] - 1)

  for i in range(len(certificate)):
    if i == col:
      certificate[i] = 1
    elif is_canonical(tableaux[:, i]):
      pivot_line = list(tableaux[:, i]).index(1)
      certificate[i] = -tableaux[pivot_line, col]

  return certificate

def solve(tableaux, previous_operations = None):
  ncons = tableaux.shape[0] - 1
  nvars = tableaux.shape[1] - 1 - ncons
  operations = np.append(np.zeros(ncons), np.identity(ncons)).reshape((ncons + 1, ncons))

  if previous_operations is not None:
    operations = previous_operations

  while not is_optimal(tableaux):
    pivot_col = [ i for i, c in enumerate(tableaux[0, :-1]) if c < 0][0]

    if is_negative(tableaux[:, pivot_col]):
      raise UnboundedLinearProgrammingError(get_unbounded_certificate(tableaux, pivot_col)[:nvars], get_solution(tableaux)[:nvars])
      
    pivot_lin = choose_pivot(tableaux, pivot_col)
    pivot_element(tableaux, operations, pivot_lin, pivot_col)

  return (tableaux, operations)

def is_viable(A, b):
  b2 = np.copy(b)
  ncons, nvars = A.shape
  aux_nvars = nvars + ncons
  A2 = np.ndarray((ncons, aux_nvars))
  A2[:, :nvars] = A
  A2[:, nvars:] = np.identity(ncons)


  for i in range(len(b)):
    if b[i] < 0:
      b2[i] *= -1
      A2[i] *= -1
  
  tableaux = build_tableaux(A2, np.zeros(aux_nvars), b2)
  tableaux[0, aux_nvars:-1] = np.ones(ncons)
  operations = operations = np.append(np.zeros(ncons), np.identity(ncons)).reshape((ncons + 1, ncons))

  for i in range(ncons):
    pivot_element(tableaux, operations, i + 1, aux_nvars + i)

  tableaux, operations = solve(tableaux, operations)
  
  return (np.isclose(tableaux[0, -1], 0), operations[0])

# Assumes that the LP is described by A, c and b in form:
#  max cx 
# s.t. Ax <= b : x > 0
def simplex(c, A, b):
  ncons, nvars = A.shape

  if not len(c) == nvars:
    raise AssertionError('Incorrect number of coeficients')

  if not len(b) == ncons:
    raise AssertionError('Incorrect number of constraints')

  is_solvable, unsolvable_certificate = is_viable(A, b)
  if not is_solvable:
    raise UnsolvableLinearProgrammingError(unsolvable_certificate)

  tableaux = build_tableaux(A, c, b)
  tableaux, operations = solve(tableaux)
  x = get_solution(tableaux)

  return (x[:nvars], tableaux[0, -1], operations[0])