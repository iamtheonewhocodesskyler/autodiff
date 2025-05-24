import numpy as np
from enum import Enum

class Operator(Enum):
   PLUS = 0
   SUB = 1
   MUL = 2
   DIV = 3
   POW = 4
   SIN = 5
   COS = 6
   TAN = 7

class Variable:
   def __init__(self, value: np.ndarray, parents=(None, None), operator=None, grad=None):
      if not isinstance(value, np.ndarray):
         value = np.array(value)

      if not np.issubdtype(value.dtype, np.number):
         raise ValueError("x must be numerical")
      
      self.value = value
      self.parents = parents
      self.operator = operator
      self.grad = grad
   
   @classmethod
   def variablify(cls, x):
      return x if isinstance(x, cls) else cls(x)
   
   
   def __add__(self, other):
      other = Variable.variablify(other)
      new = Variable(other.value+self.value)
      new.parents = (self, other)
      new.operator = Operator.PLUS
      return new
   
   def __sub__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value - other.value)
      new.parents = (self, other)
      new.operator = Operator.SUB
      return new
   
   def __mul__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value*other.value)
      new.parents = (self, other)
      new.operator = Operator.MUL
      return new
   
   def __truediv__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value/other.value)
      new.parents = (self, other)
      new.operator=Operator.DIV
      return new
   
   def __radd__(self, other):
      return self.__add__(other)
   
   def __rsub__(self, other):
      other = Variable.variablify(other)
      new = Variable(other.value-self.value)
      new.parents = (other, self)
      new.operator = Operator.SUB
      return new
   
   def __rtruediv__(self, other):
      other = Variable.variablify(other)
      new = Variable(other.value/self.value)
      new.parents = (other, self)
      new.operator = Operator.DIV
      return new
   
   def __rmul__(self, other):
      return self.__mul__(other)
   
   def __pow__(self, other):
      other = Variable.variablify(other)
      new = Variable(self.value**other.value)
      new.parents = (self, other)
      new.operator = Operator.POW
      return new
   
   def __rpow__(self, other):
      other = Variable.variablify(other)
      new = Variable(other.value**self.value)
      new.parents = (other, self)
      new.operator = Operator.POW
      return new
   
   def __repr__(self):
      parents = f"({self.parents[0].value if self.parents[0] else None}, {self.parents[1].value if self.parents[1] else None})"
      return f"Variable(value={self.value}, parents={parents}, operator={self.operator}, grad={self.grad})"
   
   def clear_gradients(self):
      self.grad = None
      a,b = self.parents
      if a is None and b is None:
         return
      if a is not None:
         a.clear_gradients()
      if b is not None:
         b.clear_gradients()

   def backward(self):
      self.clear_gradients()
      return self._backward()
   
   def __neg__(self):
      return -1 * self
   
   def _backward(self, current_chain=None):
      if current_chain is None:
         current_chain = Variable(1)
      if self.grad is None: 
         # print(self.value, self.grad)
         self.grad=current_chain
      else:
         self.grad += current_chain 

      a,b = self.parents
      if a is None and b is None:
         return
      operator = self.operator

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
         a_chain *= b * a**(b-1)
         b_chain *= self * np.log(a.value) #TODO: Change a.value to a

      if operator == Operator.SIN:
         b_chain *= cos(b)
      
      if operator == Operator.COS:
         b_chain *= -sin(b)

      if operator == Operator.TAN:
         b_chain *= 1/cos(b)**2
         

      a._backward(a_chain) if a else ...
      b._backward(b_chain) if b else ...


def sin(x):
   x = Variable.variablify(x)
   new = Variable(np.sin(x.value))
   new.parents = (None, x)
   new.operator = Operator.SIN
   return new


def cos(x):
   x = Variable.variablify(x)
   new = Variable(np.cos(x.value))
   new.parents = (None, x)
   new.operator = Operator.COS
   return new


def tan(x):
   x = Variable.variablify(x)
   new = Variable(np.tan(x.value))
   new.parents = (None, x)
   new.operator = Operator.TAN
   return new

def n_grads(x: Variable, y: Variable, n, i=1):
   y.backward()
   grad = x.grad
   if i==n:
      return [grad]
   
   n = n_grads(x,grad,n,i+1)
   n.insert(0, grad)
   return n


#Example case
if __name__ == "__main__":
   x = Variable(np.array([np.pi/2, np.pi]))
   y = x**2 + sin(x)
   ten = n_grads(x,y,10)[9]
   print(ten) #Output: -1



