import numpy as np
from enum import Enum
from time import perf_counter  

class Operator(Enum):
   PLUS = 0
   SUB = 1
   MUL = 2
   DIV = 3
   POW = 4
   SIN = 5
   COS = 6
   TAN = 7
   LOG = 8


class Variable:
   def __init__(self, value: np.ndarray, parents=(None, None), operator=None, is_constant=False, grad=None, grad_of=None, dual_numbers=(None, None), forward_mode=False, forward_wrt=None):
      #Supports both reverse and forward at the same time!
      if not isinstance(value, np.ndarray):
         value = np.array(value, dtype=np.float64)

      if not np.issubdtype(value.dtype, np.float64):
         raise ValueError("x must be numerical")
      
      self.value = value
      self.parents = parents      #BACK PARTNERS
      self.operator = operator    #BACKWARD OPERATOR
      self.grad = grad
      self.grad_of = grad_of
      # self.forward_output = forward_output
      # self.partner = partner   
      # self.partner_op = partner_op 
      self.dual_numbers = dual_numbers
      self.is_constant = is_constant
      self.forward_mode = forward_mode
      self.wrt = None
      if not forward_mode and forward_wrt:
         raise ValueError("Forward mode is set to false but 'forward_wrt' has been set. To set forward_wrt you must set forward mode to true.")
      if forward_mode and not forward_wrt:
         self.wrt = self
         self.dual_numbers = (self.value, Variable(1))
   
   @classmethod
   def variablify(cls, x):
      return x if isinstance(x, cls) else cls(x, is_constant=True)
   
   # def set_partner(self, other, operation):
   #    self.partner = other
   #    self.partner_op = operation

   #    if other: 
   #       other.partner = self 
   #       other.partner_op = operation
   
   # def set_forward(self, new, other, operation):
   #    new.operator=operation
   #    self.forward_output = new
   #    self.partner = other
   #    self.partner_op = operation

   #    if other: 
   #       other.partner = self 
   #       other.partner_op = operation

   # def get_forward_properties(self):
   #    return [self.forward_output, self.partner, self.partner_op]
   
   def _other_partial(self, other):
      # other_partial = 0
      # if other.dual_numbers != (None, None) and other.wrt==self: 
      #    other_partial = other.dual_numbers[1]
      return other.dual_numbers[1] if other.forward_mode and other.dual_numbers != (None, None) and other.wrt==self.wrt else Variable(0)     #If other has self in graph
   
   def __add__(self, other):
      other = Variable.variablify(other) 
         
      new = Variable(other.value+self.value)
      new.parents = (self, other)
      new.operator = Operator.PLUS

      if self.forward_mode:

         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         new_partial = get_partial_forward(self, other, self.dual_numbers[1], other_partial, Operator.PLUS)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True
         
      # self.set_forward(new, other, Operator.PLUS)
      # self.forward_output = new
      # self.set_partner(other)
      return new
   
   def __sub__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value - other.value)
      new.parents = (self, other)
      new.operator = Operator.SUB
      # self.set_forward(new, other, Operator.SUB)
      # self.forward_output = new
      # self.set_partner(other)
      if self.forward_mode:
         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         # new_partial = self.dual_numbers[1] - other_partial
         new_partial = get_partial_forward(self, other, self.dual_numbers[1], other_partial, Operator.SUB)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True

      return new
   
   def __mul__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value*other.value)
      new.parents = (self, other)

      new.operator = Operator.MUL
      # self.set_forward(new, other, Operator.MUL)
      # self.forward_output = new
      # self.set_partner(other)
      if self.forward_mode:
         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         # new_partial = self.value*other_partial + other.value*self.dual_numbers[1]
         new_partial = get_partial_forward(self, other, self.dual_numbers[1], other_partial, Operator.MUL)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True

      return new
   
   def __truediv__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value/other.value)
      new.parents = (self, other)
      new.operator=Operator.DIV
      # self.set_forward(new, other, Operator.DIV)
      # self.forward_output = new
      # self.set_partner(other)
      if self.forward_mode:
         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         # new_partial = (other.value*self.dual_numbers[1] - self.value*other_partial)/other.value**2
         new_partial = get_partial_forward(self, other, self.dual_numbers[1], other_partial, Operator.DIV)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True

      return new
   
   def __radd__(self, other):
      return self.__add__(other)
   
   def __rsub__(self, other):
      other = Variable.variablify(other)
      new = Variable(other.value-self.value)
      new.parents = (other, self)
      new.operator = Operator.SUB
      # self.set_forward(new, other, Operator.SUB)
      # self.forward_output = new
      # self.set_partner(other)
      if self.forward_mode:
         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         # new_partial = other_partial - self.dual_numbers[1]
         new_partial = get_partial_forward(other, self, other_partial, self.dual_numbers[1], Operator.SUB)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True

      
      return new
   
   def __rtruediv__(self, other):
      other = Variable.variablify(other)
      new = Variable(other.value/self.value)
      new.parents = (other, self)
      new.operator = Operator.DIV
      # self.set_forward(new, other, Operator.DIV)
      # self.forward_output = new
      # self.set_partner(other)
      if self.forward_mode:
         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         # new_partial = (self.value*other_partial - other.value*self.dual_numbers[1])/self.value**2
         new_partial = get_partial_forward(other, self, other_partial, self.dual_numbers[1], Operator.DIV)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True

      return new
   
   def __rmul__(self, other):
      return self.__mul__(other)
   
   def __pow__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value**other.value)
      new.parents = (self, other)
      new.operator = Operator.POW
      # self.set_forward(new, other, Operator.POW)
      # self.forward_output = new
      # self.set_partner(other)
      if self.forward_mode:
         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         # new_partial = (self.value**other.value) * (other_partial*np.log(self.value) + other.value*(self.dual_numbers[1]/self.value))
         new_partial = get_partial_forward(self, other, self.dual_numbers[1], other_partial, Operator.POW)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True

      return new
   
   def __rpow__(self, other):
      other = Variable.variablify(other)
      new = Variable(other.value**self.value)
      new.parents = (other, self)
      new.operator = Operator.POW
      # self.set_forward(new, other, Operator.POW)
      # self.forward_output = new
      # self.set_partner(other)
      if self.forward_mode:
         other_partial = self._other_partial(other)
         new.wrt = self.wrt
         # new_partial = (other.value**self.value) * (self.dual_numbers[1]*np.log(other.value) + self.value*(other_partial/other.value))
         new_partial = get_partial_forward(other, self, other_partial, self.dual_numbers[1], Operator.POW)
         new.dual_numbers = (new.value, new_partial)
         new.forward_mode = True

      return new
   
   def __repr__(self):
      parents = f"({self.parents[0].value if self.parents[0] else None}, {self.parents[1].value if self.parents[1] else None})"
      return f"Variable(value={self.value}, parents={parents}, operator={self.operator}, grad={self.grad})"
   
   def __str__(self):
      # parents = f"{self.parents[0].value if self.parents[0] else None}, {self.parents[1].value if self.parents[1] else None}"
      return f"{self.value}"
   
   def clear_gradients(self):
      self.grad = None
      if self.grad_of:
         self.grad_of.grad = Variable(0)
      # print(f"Set grad of {self} to {self.grad}")
      a,b = self.parents
      if a is None and b is None:
         return
      if a is not None:
         a.clear_gradients()
      if b is not None:
         b.clear_gradients()

   def __neg__(self):
      return -1 * self
   
   def backward(self):
      self.clear_gradients()
      return self._backward()
   
   def _backward(self, current_chain=None):
      if current_chain is None:
         current_chain = Variable(1)
      

      if self.grad is None: 
         # print(self.value, self.grad)
         self.grad=current_chain
      else:
         self.grad += current_chain 
      
      self.grad.grad_of = self

      

      a,b = self.parents
      if a is None and b is None:
         return
      operator = self.operator

      if a: 
         original_forward_A = a.forward_mode
         a.forward_mode = False #To prevent passing gradient computation to forward mode
      
      original_forward_B = b.forward_mode
      b.forward_mode = False #To prevent passing gradient computation to forward mode

      a_chain = current_chain
      b_chain = current_chain
      
      if operator == Operator.SUB:
         b_chain *= -1 
      
      if operator == Operator.MUL:
         a_chain *= b
         b_chain *= a
      
      if operator == Operator.DIV:
         a_chain *=1/b
         b_chain *= -a/(b**2)

      if operator == Operator.POW:
         a_chain *= b * utility_abs(a**(b-1))
         b_chain *= self * log(a)

      if operator == Operator.SIN:
         b_chain *= cos(b)
      
      if operator == Operator.COS:
         b_chain *= -sin(b)

      if operator == Operator.TAN:
         b_chain *= 1/cos(b)**2
      
      if operator == Operator.LOG:
         b_chain *= 1/b

      if a: a.forward_mode = original_forward_A
      b.forward_mode = original_forward_B
      a._backward(a_chain) if a else ...
      b._backward(b_chain)
   

   def in_graph(self, x):
      a,b = self.parents

      if a is x or b is x:
         return True
      
      in_a = False
      in_b = False

      if a:
         in_a = a.in_graph(x)
      
      if b:
         in_b = b.in_graph(x)

      return in_a or in_b
      
      
def get_partial_forward(x: Variable, y: Variable, dx, dy, operator: Operator):
   """x is right, y is left"""
   # if x: x=x.value
   # y=y.value
   if x: 
      original_x = x.forward_mode
      x.forward_mode = False

   original_y = y.forward_mode
   y.forward_mode = False

   if operator == Operator.PLUS:
      dz = dx+dy
   elif operator == Operator.SUB:
      dz = dx-dy
   elif operator == Operator.MUL:
      dz = x*dy + y * dx
   elif operator == Operator.DIV: # x/y
      dz = (y*dx - x*dy)/(y**2)
   elif operator == Operator.POW: # x^y
      dz = (x**y) * (dy*log(x) + y*(dx/x))
   elif operator == Operator.SIN: #cos(y)
      dz = cos(y) * dy
   elif operator == Operator.COS:
      dz = -sin(y) * dy
   elif operator == Operator.TAN:
      dz = (1/cos(y)**2) * dy
   elif operator == Operator.LOG:
      dz = dy/y

   if x: x.forward_mode = original_x
   y.forward_mode = original_y
   
   return dz

   

def sin(x):
   x = Variable.variablify(x)
   new = Variable(np.round(np.sin(x.value), 15))
   new.parents = (None, x)
   new.operator = Operator.SIN
   # x.set_forward(new, None, Operator.SIN)
   # x.forward_output = new
   # x.set_partner(None)
   if x.forward_mode:
      # other_partial = x._other_partial(other)
      new.wrt = x.wrt
      # new_partial = np.cos(x.value)*x.dual_numbers[1]
      new_partial = get_partial_forward(None, x, None, x.dual_numbers[1], Operator.SIN)
      new.dual_numbers = (new.value, new_partial)
      new.forward_mode = True

   return new


def cos(x):
   x = Variable.variablify(x)
   new = Variable(np.round(np.cos(x.value), 15))
   new.parents = (None, x)
   new.operator = Operator.COS
   # x.set_forward(new, None, Operator.COS)
   # x.forward_output = new
   # x.set_partner(None)
   if x.forward_mode:
      # other_partial = x._other_partial(other)
      new.wrt = x.wrt
      # new_partial = -np.sin(x.value)*x.dual_numbers[1]
      new_partial = get_partial_forward(None, x, None, x.dual_numbers[1], Operator.COS)
      new.dual_numbers = (new.value, new_partial)
      new.forward_mode = True

   return new


def tan(x):
   
   x = Variable.variablify(x)
   new = Variable(np.tan(x.value))
   new.parents = (None, x)
   new.operator = Operator.TAN
   # x.set_forward(new, None, Operator.SIN)
   # x.forward_output = new
   # x.set_partner(None)
   if x.forward_mode:
      # other_partial = x._other_partial(other)
      new.wrt = x.wrt
      # new_partial = 1/np.cos(x.value)**2 * x.dual_numbers[1]
      new_partial = get_partial_forward(None, x, None, x.dual_numbers[1], Operator.TAN)
      new.dual_numbers = (new.value, new_partial)
      new.forward_mode = True

   return new

def log(x: Variable):
   x = Variable.variablify(x)
   new = Variable(np.log(x.value))
   new.parents = (None, x)
   new.operator = Operator.LOG

   if x.forward_mode:
      # other_partial = x._other_partial(other)
      new.wrt = x.wrt
      # new_partial = 1/np.cos(x.value)**2 * x.dual_numbers[1]
      new_partial = get_partial_forward(None, x, None, x.dual_numbers[1], Operator.LOG)
      new.dual_numbers = (new.value, new_partial)
      new.forward_mode = True
   
   return new

def utility_abs(x: Variable): #Not to be used in computational graphs, only for handling numerical errors. A seperate function to be used in computational graph.
   x = Variable.variablify(x)
   x.value = np.abs(x.value)

   return x
   

#REVERSE MODE N-TH GRAD
def n_grads_rev(x: Variable, y: Variable, n, i=1): #Gives a list of gradients from first order to nth order. Does NOT include 0th derivative.
   y.backward()
   grad = x.grad
   
   if grad.value == -0.0: grad.value = 0.0 #Fixing numerical error
   if not grad.in_graph(x):
      return [grad]*(n+1)
   if i==n:
      return [grad]
   
   n = n_grads_rev(x,grad,n,i+1)
   n.insert(0, grad)
   return n

# def n_grad(x: Variable, y: Variable, n):    TODO: Implement n_grad via forward mode


#Example case
if __name__ == "__main__":
   def f(x):
      return sin(log(x**2 + 1)) + cos(x**3)*tan(2*x)
   x = Variable(np.pi/2, forward_mode=False)
   y = f(x)
   y.backward()
   # x.forward_mode=False
   # y.forward_mode=True

   # print(y.dual_numbers[1])
   print(x.grad)
   # print([i.value for i in n_grads_rev(x, y, 10)])

   # i = perf_counter()
   # y.backward()
   # f = perf_counter()
