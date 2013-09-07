import sympy, sympy.abc
import copy
from sympy.utilities.lambdify import lambdify
import sfepy.physics.radial_mesh
import scipy.special

class BaseFce(object):
      def __init__(self, xmul = 1.0, xshift = 1.0, ymul = 1.0):
          self.xmul = xmul
          self.xshift = xshift
          self.ymul = ymul
      
      def n_derivation(self, n, x):
          if n==0: return self(x)
          if n==1: return self.derivation(x)
          if n==2: return self.second_derivation(x)
          raise Exception('Derivation grade to high')
          
      def __call__(self, x):
          if isinstance(x, sfepy.physics.radial_mesh.BaseRadialMesh):
             x=x.coors
          return self.call(x * self.xmul + self.xshift) * self.ymul 
      
      def derivation(self, x):
          return self.derivate_1(x * self.xmul + self.xshift) * self.xmul * self.ymul
          
      def second_derivation(self, x):
          return self.derivate_2(x * self.xmul + self.xshift) * self.xmul * self.ymul
          
      def scale(self, scale):
          out = copy.copy(self)
          out.ymul = out.ymul * scale
          return out
          
      
 

class Basis(BaseFce):
      def __init__(self, base, xmul = 1.0, xshift = 0.0, ymul = 1.0):
          self.base = base
          BaseFce.__init__(self, xmul, xshift, ymul)
      
      def call(self, x):
          return reduce(lambda o,y: o+y(x), self.base, 0.0) 
          
      def derivation(self, x):
          return reduce(lambda o,y: o+y.derivation(x), self.base, 0.0)
          
      def second_derivation(self, x):
          return reduce(lambda o,y: o+y.second_derivation(x), self.base, 0.0)
          
      def __iter__(self):
          return iter(self.base)
      
      def __len__(self):
          return len(self.base)
          
      def __getitem__(self, x):
          return self.base[x]
          
      @property   
      def size(self):
          return len(self.base)
          
      def scale(self, scale):
          if isinstance(scale, (int, float)):
             return BaseFce.copy(scale)
          base = [ b.scale(x) for b,x in zip(self.base, scale)]
          out = copy.copy(self)
          out.base = base
          return out          
          
class SympyFce(BaseFce):
      default_functions = {
	    	'erf' : scipy.special.erf
	    }
	  
      def __init__(self, fce, xmul = 1.0, xshift =0.0, ymul = 1.0):
          BaseFce.__init__(self, xmul, xshift, ymul)
          self.functions = self.default_functions
          self.expr = sympy.sympify(fce)
          self.fce = None
          self.d1 = None
          self.d2 = None

      def __str__(self):
          return "SympyFce: %s" % self.expr
          
      def __repr__(self):
          return str(self.expr)
          
      def call(self, x):
          if self.fce is None:
          	 self.fce = lambdify(sympy.abc.x, self.expr, ["numpy", self.functions ] )          
          return self.fce(x)
          
      def derivate_1(self, x):
          if self.d1 is None:
            self.expr_d1 = self.expr.diff()
            self.d1 = lambdify(sympy.abc.x, self.expr_d1, ["numpy", self.functions ])
          return self.d1(x)          
            
      def derivate_2(self, x):
          if not self.d2:
            self.derivate_1(0.0)
            self.expr_d2 = self.expr_d1.diff()
            self.d2 = lambdify(sympy.abc.x, self.expr_d2, ["numpy", self.functions ])
          return self.d2(x)
            
      def __getattr__(self, name):
          sm = getattr(self.expr, name)
          args = [a.expr if isinstance(a, SympyFce) else a]
          return lambda *args: SympyFce(sm(*args))
