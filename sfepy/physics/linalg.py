import numpy as np

def orthogonalize(vectors, scalar_product = None, b_operator = None, reortogonalization =0, out = None, orthogonalized = 0, empty_vectors = False):
      """ Orthogonalize sets of vector. It can be done by standard or user defined scalar product.
          If b_operator is given, B-orthogonalization using this b_operator is performed. In this mode the
          scalar_product can be negative definite, so the function result has always three parts:
           V = orthogonalized set of vectors (numpy 2d array)
           C = vector of coeficients 1 or -1, that says, whether b_operator on given vector is negative or positive definite
           W = b_operator * set of vectors (numpy 2d array) 
           So the b_operator can be expressed in separable form as W^CWa

          Result will be in newly allocated buffer or function can use buffers given in out. 
          In orthogonalized parametr one can state, that there are in the out array #orthogonalized
           just allready orthogononalized vectors, vectors given as input will be then orthogonalized
           even against that vectors and will be appended after that vectors.
      """

      dim = len(vectors[0])
      if out is None:
         c_coef = np.ones(len(vectors), dtype=np.double)
         vorto = np.empty( (len(vectors), dim) )
         orto = np.empty( (len(vectors), dim) )
      else:
         orto = out[0]
         c_coef = out[1]
         c_coef[orthogonalized:] = 1
         vorto = out[2]
      
      if scalar_product is None:
         scalar_product = lambda x,y: sum(x * y)

      if isinstance(b_operator, np.ndarray):
         b_vector = b_operator
         b_operator = lambda x: b_vector * x

      for reorto in xrange(0, reortogonalization+1):
          r = orthogonalized
          for s in xrange(0, len(vectors)):
                base = vectors[s]
                for t in xrange(0, r):
                    base -= orto[t] * c_coef[t] * scalar_product(base, vorto[t])
                vbase = b_operator(base) if b_operator else base
                scale = scalar_product(base, vbase)
                if scale < 0:
                    c_coef[r] = -1
                    scale = -scale
                elif scale == 0.0:
                    if empty_vectors:
                       orto[r] = 0.0
                       c_coef[r] = 0
                       r+=1
                    continue
                    
                scale = np.sqrt(scale)
                orto[r] = base / scale
                if b_operator:
                   vorto[r] = vbase / scale
                r+=1
          #use output as input      
          vectors = orto[orthogonalized:]

      c_coef[r:] = 0.0
      if not b_operator:
         vorto = None
      return orto, c_coef, vorto


