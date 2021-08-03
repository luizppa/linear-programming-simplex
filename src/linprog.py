# -*- coding: utf-8 -*-
"""
Simplex Algorithm
"""

import numpy as np
import math

"""
- General purpose functions and classes

- Unbounded and unsolvable LPs

We will be using exceptions to handle unbounded and unsolvable linear programming problems.
Both cases will raise an exception that take as an argument an array representing it's respective certificate.
"""
class LinearProgrammingError(Exception):
  def __init__(self, *args):
    self.message = args[0]
    self.certificate = args[1]

  def __str__(self):
    return '{0}, certificate = {1}'.format(self.message, self.certificate)

class UnboundedLinearProgrammingError(LinearProgrammingError):
  def __init__(self, *args):
    super(UnboundedLinearProgrammingError, self).__init__('Optimal value is infinite', args[0])

class UnsolvableLinearProgrammingError(LinearProgrammingError):
  def __init__(self, *args):
    super(UnsolvableLinearProgrammingError, self).__init__('Linear programming has no solution', args[0])

"""
- Utilitary methods

A few helper methods will auxiliate our algorithm such as:

* compare(a, symbol, b):              where symbol can be ==, >, <, >= or <=. Returns whether the relation holds true considering a tolerance.
* is_optimal(tableaux):               returns True if the tableaux has no negative c_i
* is_unbounded(tableaux, pivot_col):  returns True if a column of A has all negative entries
* is_canonical(arr):                  returns True if a arr is a vector containing only 0's and 1's and the sum of it's
  -                                   components is equal to 1 (any array in form [0\*, 1, 0\*]).
"""
def equal(a, b):
  return np.isclose(a, b)

def greater_than(a, b):
  return a > b and not equal(a, b)
  
def smaller_than(a, b):
  return a < b and not equal(a, b)
  
def greater_or_equal(a, b):
  return not smaller_than(a, b)
  
def smaller_or_equal(a, b):
  return not greater_than(a, b)

def compare(a, symbol, b):
  if symbol == '==':
    return equal(a, b)
  elif symbol == '>':
     return greater_than(a, b)
  elif symbol == '<':
     return smaller_than(a, b)
  elif symbol == '>=':
     return greater_or_equal(a, b)
  elif symbol == '<=':
     return smaller_or_equal(a, b)

def is_optimal(tableaux):
  return all(compare(c, '>=', 0) for c in tableaux[0, :-1])

def is_negative(arr):
  return all(compare(a, '<', 0) for a in arr)

def is_canonical(arr):
  sum = 0

  for i in range(len(arr)):
    if not equal(arr[i], 1) and not equal(arr[i], 0):
      return False
    sum += arr[i]
  
  return equal(sum, 1)

"""
- The specifics

- Building the tableaux

The following cells are responsible for the manipulation of the tableaux used to solve the simplex. 

build_tableaux(A, c, b) receives the matrix of constraints A and the vector b where Ax = b on the
linear programming and the vector of coeficients c, then proceeds to building the tableaux consisting of:

$$
\begin{bmatrix}
    -c_{1} & -c_{2} & \dots  & c_{n} & 0 & \dots & 0 & 0 \\
    A_{11} & A_{12} & \dots  & A_{1n} & 1 & \dots & 0 & b_{1} \\
    \vdots & \vdots & \ddots & \vdots & 0 & \ddots & 0 & \vdots \\
    A_{m1} & A_{m2} & \dots  & A_{mn} & 0 & \dots & 1 & b_{m}
\end{bmatrix}
$$

where m is the number of constraints and n is the number o variables of interest.
"""
def build_tableaux(A, c, b):
  ncons, nvars = A.shape
  cn = [ -i for i in c ]

  tableaux = np.zeros((ncons + 1, nvars + ncons + 1))
  tableaux[1:, :nvars] = A
  tableaux[1:, nvars:-1] = np.identity(ncons)
  tableaux[0, :-1] = np.append(cn, np.zeros(ncons))
  tableaux[1:, -1] = b
  return tableaux

"""
- Simplex helpers

During the execution of each iteration of the simplex, we must choose an element from a column that represents
a variable that can potentially increase the value of the objective function to be the pivot of that iteration.
To do so, we must choose the element of given column j that satisfies:

$$
t = min \begin{Bmatrix} \frac{b_i}{A_{ij}} : A_{ij} > 0 \end{Bmatrix}
$$

given the tableaux and the column j, choose_pivot(tableaux, col) returns the index i of t.

After obtaining the index to the pivot, pivot_element(tableaux, operations, lin, col) is responsible for performing
the pivoting operations over the tableaux and register all of those operations on our operations registry matrix.

Finally, get_solution(tableaux) extracts the value of the variables of interest from the solved tableaux.
On the event of an unbounded LP, given it's tableaux and a column col representing the variable that makes that LP unbounded,
get_unbounded_certificate(tableaux, col) retrieves a certificate of unboundness for the LP.
"""
def choose_pivot(tableaux, col):
  values = np.zeros(tableaux.shape[0] - 1)
  
  for i in range(1, tableaux.shape[0]):
    if compare(tableaux[i, col], '>', 0):
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

"""
- Simplex execution

This implementation reffers to a two phase simplex algorithm, therefore we first
build and solve an auxiliary LP to determine wheter the origin LP is viable or unviable.

- Solving a known viable LP

The solve(tableaux, [previous_operations = None]) function creates an operations registry matrix
and performs pivoting operations over the received tableaux until it is in it's optimal form.
It can optionally receive a previous_operations matrix that is also a operations registry containing
a set of operations already performed over that tableaux, this option is used during the first phase
of the simplex where we need to solve an auxiliary LP.
"""
def solve(tableaux, previous_operations = None):
  ncons = tableaux.shape[0] - 1
  nvars = tableaux.shape[1] - 1 - ncons
  operations = np.append(np.zeros(ncons), np.identity(ncons)).reshape((ncons + 1, ncons))

  if previous_operations is not None:
    operations = previous_operations

  while not is_optimal(tableaux):
    pivot_col = [ i for i, c in enumerate(tableaux[0, :-1]) if compare(c, '<', 0)][0]

    if is_negative(tableaux[:, pivot_col]):
      raise UnboundedLinearProgrammingError(get_unbounded_certificate(tableaux, pivot_col)[:nvars], get_solution(tableaux)[:nvars])
      
    pivot_lin = choose_pivot(tableaux, pivot_col)
    pivot_element(tableaux, operations, pivot_lin, pivot_col)

  return (tableaux, operations)

"""
- Checking if a LP is viable

However, prior to using the solve function, we must determine the viability of our LP, thus, we build an auxiliary LP
that is guaranteed to be viable and that can indicate one of two cases:

* it's optimal value is 0, in which case the original LP is viable
* it's optimal value is less than 0, in which case, our original LP is unviable

For that we build a tableaux of form:

$$
\begin{bmatrix}
    0 & 0 & \dots  & 0 & 1 & \dots & 1 & 0 \\
    A^{'}_{11} & A^{'}_{12} & \dots  & A^{'}_{1n} & 1 & \dots & 0 & b^{'}_{1} \\
    \vdots & \vdots & \ddots & \vdots & 0 & \ddots & 0 & \vdots \\
    A^{'}_{m1} & A^{'}_{m2} & \dots  & A^{'}_{mn} & 0 & \dots & 1 & b^{'}_{m}
\end{bmatrix}
$$

where

$$
b^{'}_{i} = \begin{cases}
  b_{i} & \text{if } & b_{i} \geq 0 \\
  -b_{i} & \text{otherwise}
\end{cases}
$$

and 

$$
A^{'}_{ij} = \begin{cases}
  A_{ij} & \text{if } & b_{i} \geq 0 \\
  -A_{ij} & \text{otherwise}
\end{cases}
$$

then we register the transofrmations made in A' and b' in an operations registry matrix and solve the tableaux.
"""
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
  
  return (equal(tableaux[0, -1], 0), operations[0])

"""
- Solving with simplex

Finally, we can use simplex(c, A, b) to wrap al the auxiliary functions and solve our LP.
The simplex will verify a few constraints that are required for it to work (basically verify if the LP is well formated),
then check if the provided LP is viable, if so, it proceeds to solving it.
The function returns a 3-tuple containing the solution vector for x, the optimal value and the certificate of the LP.
"""
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