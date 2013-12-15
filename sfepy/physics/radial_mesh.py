import math
from sfepy.physics.basis import SympyFce, Basis
from scipy.integrate import simps, cumtrapz
from sfepy.linalg import norm_l2_along_axis
import scipy.interpolate as si
import scipy
import numpy as np
import copy
import types
import inspect
from sfepy.physics.linalg import orthogonalize
from scipy.optimize import curve_fit

class RadialVector(object):

    """ Read array of radial vectors from file """
    @staticmethod
    def from_file(file):
        array = np.genfromtxt(file)
        mesh = RadialMesh(array[:, 0])
        return [RadialVector(mesh, array[:, r])
                for r in xrange(1, array.shape[1])]

    def brother(self, values = None):
        if values is None:
           values = np.zeros_like(self.values)
        return RadialVector(self.mesh, values)

    def __repr__(self):
        return self.pretty(10)

    def __abs__(self):
        return RadialVector(self.mesh, np.abs(self.values))

    def __pow__(self, exponent):
        return RadialVector(self.mesh, self.values ** exponent)

    def __eq__(self,  vector):
        if isinstance(vector, RadialVector):
           if vector.mesh == self.mesh:
              return (vector.values == self.values).all()
           else:
              return (vector.interpolate(self.mesh.coors) == self.values).all()
        else:
           vector = np.asarray(vector)
           return vector.shape == self.values.shape and (vector == self.values).all()

    def scale(self, scale = None, shift = None):
        self.clear_cache()
        self.mesh = self.mesh.copy()
        self.mesh.scale(scale, shift)
        self.parent_mesh = None
        self.midpoint_mesh = None

    def copy(self):
        out = copy.copy(self)
        out.values = out.values.copy()
        return out

    def pretty(self, values):
        """ Pretty print <values> values of vector
        @param int number of values printed
        """
        size = self.values.size
        if values > size:
            n = np.arange(size, dtype=np.int)
        else:
            n = np.array(np.linspace(0, self.mesh.size - 1, values),
                         dtype=np.int)
        out = 'RadialVector: '
        for x in n[0:-1:1]:
            out += '%f: %g,  ' % (self.mesh.get_r(x), self.values[x])
        x = n[-1]
        out += '%f: %g' % (self.mesh.get_r(x), self.values[x])
        return out

    def __init__(self, mesh, values=None):
        if not isinstance(mesh, RadialMesh):
            mesh = RadialMesh(mesh)
        self.mesh = mesh
        if isinstance(values, types.FunctionType) or hasattr(values, '__call__'):
           values = values(mesh.coors)
        self.values = (np.asarray(values, np.float) if values
                       is not None else np.zeros(self.mesh.size))
        self.clear_cache()

    def clear_cache(self):
        self.derivate_cache = {}
        self.interpolator = {}
        self.extrapolated = None
        self.precision = None
        self.result_precision = None
        self.running = None

    def running_mean(self):
        """ Smooths vector """
        if self.running is None:
            weights = np.array((1.0, 3, 4, 3, 1))
            weights = weights / weights.sum()
            wsize = int(weights.size - 1)
            data = np.hstack([np.ones(wsize) * self.values[0], self.values,
                              np.ones(wsize / 2) * self.values[-1]])
            self.running = np.convolve(data, weights)[wsize:-(wsize / 2)
                                                      - wsize]
        return self.running

    def integrate(self, factor = None):
        """ Compute integral of vector with given integral factor.
            See RadialMesh.integrate for possible factors.
        """
        return self.mesh.integrate(self.values, factor)

    def linear_integrate(self):
        """ Compute (not-spherical) integral of vector """
        return self.mesh.linear_integrate(self.values)


    def get_extrapolated(self, precision=0.0001, grade=10, attempts=10):
        """ Try to smooth and extrapolate curve trough vector points with given precision,
        if can't be done, it raise the precision
        @param precision int max error in points
        @param grade int grade of spline
        @param attempts max number of tries to gues the precision
        """

        if precision is None:
            return si.InterpolatedUnivariateSpline(self.mesh.get_coors(),
                                                   self.values, k=5)

        if ((self.extrapolated is None)
            or (self.precision == self.result_precision)
            and (precision < self.precision)):

            self.precision = precision
            data = self.running_mean()
            for attempt in xrange(attempts):
                self.extrapolated = si.UnivariateSpline(self.mesh.coors,
                                                        data, k=5, s=precision)
                der = self.extrapolated(self.mesh.coors, 1)
                sig = np.sign(der)
                if np.abs(sig[1:] - sig[:-1]).sum() <= grade:
                    break
                precision = precision * 2
        self.result_precision = precision
        a = self.extrapolated
        return a

    def extrapolated_values(self, at=None, precision=0.0001, grade=10,
                            attempts=10):
        """ Smoth the vector by interposing spline curve and return smoothed values """
        if at is None:
            at = self.mesh
        elif not isinstance(at, RadialMesh):
            at = RadialMesh(at)
        val = self.get_extrapolated(precision, grade, attempts)(at.coors)
        return RadialVector(at, val)

    def extrapolated_derivatives(self, at=None, precision=0.0001, attempts=10):
        """ Smoth the vector by interposing spline curve and return derivatives """
        if at is None:
           at = self.mesh
        elif not isinstance(at, RadialMesh):
            at = RadialMesh(at)
        val = self.get_extrapolated(precision=0.0001, grade=10,
                                    attempts=10)(at.coors, 1)
        return RadialVector(at, val)

    def derivation(self, factor = None, from_zero = False, at = None, force=False):
        """ Return radial vector of derivatives and if 'at' is not none return derivatives
            in given points
            Can derivate with respect to given integral factor, see mesh.integrate
        """
        key = (factor or self.mesh.integral_factor) + str(from_zero)
        if not self.derivate_cache.has_key(key) or force:
           self.derivate_cache[key] = self.mesh.derivation(self.values, factor, from_zero)
        if at is not None:
           return self.derivate_cache[key].interpolate(at)
        return self.derivate_cache[key]

    def linear_second_derivation(self,  from_zero = False, at = None):
        """
        Return vector from second derivations of given vector
        .. math::
        """
        return self.second_derivation( 'linear', from_zero, at)

    def second_derivation(self, factor = None, from_zero = False, at = None):
        """
        Return vector from second derivations of given vector that was integrated using given integral factor
        .. math::
        """
        return self.mesh.second_derivation(self, factor, from_zero, at)

    def linear_derivation(self, from_zero = False, at=None, force = False):
        """ Return radial vector of derivatives and if 'at' is not none return derivatives
            in given points.
            Linear derivation (no integration factor).
        """
        return self.derivation(at = at, factor = 'linear', force = force, from_zero = from_zero)

    def slice(self, x, y=None):
        """ Return slice of vector, given by two indexes (start, stop+1) or
            float bounds <a, b) """
        if isinstance(x, float):
            x = self.mesh.get_index(x)
        elif x is None:
            x=0
        if isinstance(y, float):
            y = self.mesh.get_index(y)
        elif y is None:
            y = self.mesh.size
        return RadialVector(self.mesh.slice(x, y), self.values[x:y])

    def at(self, x, out=None):
        return self.interpolate(x, out = out)

    def interpolate(self, x, kind = None, centre = None, out=None):
        """ Return values interpolated in given coordinates x.
        Kind of interpolation can be None for linear interpolation or
        some of scipy.interpolate.interp1d kinds.

        Caches computed interpolator for non-linear cases.
        If centre is not None, coordinates are 3d and the resulting coordinates
        are the distances from the given centre.
        """
        if kind is None:
           return self.mesh.interpolate(self.values, x, centre, kind, out)
        if not self.interpolator.has_key(kind):
           self.interpolator[kind]=self.mesh.interpolator(self.values, kind)
        if(centre is not None):
           if centre.any():
              x = x - centre
           x = norm_l2_along_axis(x, axis=1)
        if out:
           out[:] = self.interpolator[kind](x)
           return out
        return self.interpolator[kind](x)

    def interpolate_3d(self, x, kind = None, centre = None, out = None):
        """ Return interpolated values at points given by 3d coordinates and centre """
        if centre is None:
          centre = np.zeros((3,))
        return self.interpolate(x, kind, centre, out)

    def output_vector(self, filename=None, other=None):
        """ Save vector in two columns COORS VALUES """
        return self.mesh.output_vector(self, filename, other)

    @staticmethod
    def sparse_merge(vectors):
        mesh = RadialMesh.merge([v.mesh for v in vectors])
        return [mesh.sparse_vector(v) for v in vectors]

    def _get_values_from_object(self, data):
        return data if isinstance(data, np.ndarray) else \
               data(self.mesh) if isinstance(data, RadialVector) else \
               data(self.coors) ;

    def __iadd__(self, vector):
        if(isinstance(vector, (int, float))):
           self.values += vector
        else:
          values = self._get_values_from_object(vector)
          self.values +=values
        return self

    def __add__(self, vector):
        if(isinstance(vector, (int, float))):
          return RadialVector(self.mesh, self.values + vector)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, self.values + values)

    def __isub__(self, vector):
        if(isinstance(vector, (int, float))):
           self.values -= vector
        else:
          values = self._get_values_from_object(vector)
          self.values -=values
        return self

    def __sub__(self, vector):
        if(isinstance(vector, (int, float))):
          return RadialVector(self.mesh, self.values - vector)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, self.values - values)

    def __radd__(self, vector):
        if(isinstance(vector, (int, float))):
           return RadialVector(self.mesh, self.values + vector)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, self.values + values)

    def __rsub__(self, vector):
        if(isinstance(vector, (int, float))):
           return RadialVector(self.mesh, vector - self.values)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, values - self.values)

    def __mul__(self, vector):
        if(isinstance(vector, (int, float))):
           return RadialVector(self.mesh, self.values * vector)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, self.values * values)

    def __rmul__(self, vector):
        if(isinstance(vector, (int, float))):
           return RadialVector(self.mesh, self.values * vector)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, self.values * values)

    def __imul__(self, vector):
        if(isinstance(vector, (int, float))):
           self.values *= vector
        else:
           values = self._get_values_from_object(vector)
           self.values *=values
        return self

    def __div__(self, vector):
        if(isinstance(vector, (int, float))):
           return RadialVector(self.mesh, self.values / vector)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, self.values / values)

    def __idiv__(self, vector):
        if(isinstance(vector, (int, float))):
          self.values /= vector
        else:
          values = self._get_values_from_object(vector)
          self.values /=values
        return self

    def __ipow__(self, vector):
        if(isinstance(vector, (int, float))):
           self.values **= vector
        else:
          self.values = self._get_values_from_object(vector)
          self.values **=vector
        return self

    def __rdiv__(self, vector):
        if(isinstance(vector, (int, float))):
           return RadialVector(self.mesh, vector / self.values)
        else:
          values = self._get_values_from_object(vector)
          return RadialVector(self.mesh, values / self.values)

    def __invert__(self):
        return self.derivation()

    def __getitem__(self, r):
        return self.values[r]

    def __setitem__(self, r, val):
        self.values[r] = val

    def plot(self, range=None, other=None):
        """ Plot vector using given shell script """
        return self.mesh.plot(self, range=range, other=other)

    def get_coors(self):
        """ Return mesh coors """
        return self.mesh.coors

    def __call__(self, at):
        """ Call numpy functions on vector """
        if self.mesh == at or at is self.mesh.coors:
           return self.values
        if isinstance(at, RadialMesh):
           at = at.coors
        return self.interpolate(at)

    def __getattr__(self, name):
        """ Read numpy array attributes """
        return getattr(self.values, name)

    def partition(self, block=None):
        """
        Split again vector joined from more radial vectors.
        The mesh.coors should look like a saw, it's divided to single tooths.
        """
        meshes = self.mesh.partition()
        out = []
        start = 0
        for m in meshes:
            size = m.size
            out.append(RadialVector(m, self.values[start:start + size]))
            start += size
        if block is not None:
           out=out[block]
        return out

    def norm(self, factor = None):
        return math.sqrt(self.dot(self, factor))

    def dot(self, vector, factor = None):
        """ Return dot product with given vector using given integral factor """
        return self.mesh.dot(self.values, vector, factor)

    def v_dot(self, a, b, factor = None):
        """ Return dot product of two vectors in self-norm using given integral factor
            .. math::
               (a, b)_{vector} = \int a * (b * vector)' * factor dx
        """
        return self.mesh.dot(a, self.values*b, factor)

    def integral(self, factor = None, from_zero=False):
        return self.mesh.integral(self.values, factor, from_zero)

    def linear_integral(self, from_zero=False):
        """ Compute running integral of vector. See mesh.linear_integral """
        return self.mesh.integral(self.values, 'linear', from_zero)

    def normalize(self, factor = None):
        self.values /= self.norm(factor)
        return self

    def bind_to_zero(self, start, end, base = None, factor = 'linear', endpoint = (0.0,0.0,0.0), eps = 0.9):
        start, startr = self.mesh.get_index_r(start)
        end, endr = self.mesh.get_index_r(end)

        if base == None:
           base = [
                   SympyFce("x*x*exp(-x*x)",         1/startr),
                   SympyFce("x*x*x*x*exp(-x*x)",     1/startr),
                   SympyFce("x*x*x*x*exp(-x*x*x*x)", 1/startr),
                   SympyFce("x*x*exp(-x*x)",         1/endr),
                   SympyFce("x*x*x*x * exp(-x*x)",     1/endr),
                   SympyFce("x*x*x*x*exp(-x*x*x*x)", 1/endr)
                  ]
        if not(isinstance(base, Basis)):
           base = Basis(base)

        dim =base.size
        mat = np.empty((dim, dim))
        right = np.empty((dim))
        d1 = self.derivation(factor = factor)
        d2 = self.second_derivation(factor = factor)
        point = startr

        for x in xrange(dim):
            here = point
            if int(x / 3) == 0:
               right[x] = endpoint[x]
               here = endr
            elif x % 3 ==0:
               right[x] = self(point)
            elif x % 3 ==1:
               right[x] = d1(point)
            elif x % 3 ==2:
               right[x] = d2(point)
               point = startr * eps if point == startr else (startr + point)/2
            for y in xrange(dim):
                mat[x,y] = base[y].n_derivation(x % 3, here)

        par=scipy.linalg.solve(mat, right)
        scaled = base.scale(par)
        self.values[start:end] = scaled(self.mesh[start:end])
        self.values[end:] = 0.0
        return self


    def smooth_to_zero_from(self, start, end):
        """
        Modify its values, so it wil begin fall to zero at start
        and from end they will be zero.
        """
        start, startr = self.mesh.get_index_r(start)
        end, endr = self.mesh.get_index_r(end)
        value = self[start]
        inter = self.slice(0, end)

        data = np.array([[value,
                          inter.extrapolated_derivatives(startr,
                                                         precision=None).values],
                         [0.0, 0.0]])

        curve = si.PiecewisePolynomial([startr, endr], data)
        coors = self.mesh.get_coors()

        self.values[start:end + 1] = curve(coors[start:end + 1])
        self.values[end + 1:] = 0.0
        return self

    def smooth_to_zero(self, at_radius, min_ratio = 0.6, smoothness = 0.9):
        rci = self.mesh.get_index(at_radius)
        avalues = np.absolute(self.values)

        compare_with = 1.0e200
        count = 0

        mini = self.mesh.get_index(at_radius * min_ratio)
        for i in xrange(rci, mini, -1):
            if avalues[i] >= compare_with:
                count += 1

            else:
                if count >= 4:
                    i = i + 1
                    break
                count = 0
                compare_with = avalues[i]
                sloopstart = i

        if sloopstart <= mini + 1:
            sloopstart = self.mesh.get_index(at_radius * 0.95)
            i = self.mesh.get_index(at_radius * 0.95 * smoothness)

        else:
            limit = self.mesh.get_r(sloopstart) * smoothness
            if limit > self.mesh.get_r(i):
                i = self.mesh.get_index(limit)

        return self.smooth_to_zero_from(i, rci)

    def find_zero_r(self):
        for r in xrange(self.mesh.size-2, -1, -1):
            if self[r] != 0.0:
               return self.mesh[r+1]
        return 0

    def space_derivation(self, dimension = 3, from_zero = False, at = None):
        return self.mesh.space_derivation(self, dimension, from_zero, at)

    def piecefit(self, f=None, step=None, minval=None, initial = None):
      x=self.mesh.coors
      y=self.values
      if f is None:
         f = lambda x,a,b,c,d,e,f:\
             ((a*x+b)*x+c)*x+d + x*x*np.exp(f*(x-e)**2)
         f = lambda x,a,b,c,d:\
             ((a*x+b)*x+c)*x+d

      if minval is None:
         minval = len(inspect.getargspec(f)[0])

      if step is None:
         step = (x[-1] - x[0])/len(x) * minval * 4

      out = np.zeros(len(x))
      halfstep = step / 2

      def aprox(frm, to):
          tto = to
          ffr = frm
          if tto - frm < minval:
             add = minval - tto + ffr
             tto += add / 2
             ffr -= add - add / 2
          if tto > len(x):
             s = tto - len(x)
             tto = len(x)
             ffr -= s
          if ffr < 0:
             tto -= ffr
             ffr = 0
          res, _ = curve_fit(f, x[ffr:tto], y[ffr:tto], initial)
          return f(x[frm:to], *res)

      def find_limit(frm, limit):
          while frm < len(x) and x[frm] < limit:
                frm+=1
          return frm
      i = 0
      start = x[0] + halfstep
      oi = find_limit(i, start)
      ii = find_limit(i, start+halfstep)
      out[i:ii] = aprox(i,ii)
      out[oi:ii] *= (1 - np.abs(start - x[oi:ii]) / halfstep)
      i = oi
      while start < x[-1] - step:
            start += halfstep
            oi = i
            i = find_limit(i, start)
            ii = find_limit(i, start + halfstep)
            if ii == oi:
                continue
            out[oi:ii] += aprox(oi,ii) *  (1 - np.abs(start - x[oi:ii]) / halfstep)
      start += halfstep
      ii = find_limit(i, start)
      vals = aprox(i,len(x))
      out[i:ii] +=  vals[:ii-i] *  (1 - np.abs(start - x[i:ii]) / halfstep)
      out[ii:]  = vals[ii-i:]
      return self.brother(out)

class BaseRadialMesh(object):
    """
    Radial mesh.
    """

    def _get_values_from_object(self, data):
        """
        return values used for computation, allow use RadialVector or lambda function as input
        in funcitons
        """
        return data if isinstance(data, np.ndarray) else \
               data(self) if isinstance(data, RadialVector) else \
               data(self.coors) ;

    def mul_integral_factor(self, vector, factor):
        """
        return values multiplied by given integral factor
        It can be
        'linear' - no integral factor
        'spehrical' - spherical integral factor: 4*math.pi * r**2
        'r2' - physics spherical integral factor (used e.g. with spherical harmonics): r**2
        'd2' - used when derivate 2D space integral to get radial function
        'd3' - used when derivate 3D space integral to get radial function
        """
        vector = self._get_values_from_object(vector)
        factor = factor or self.integral_factor
        if factor == 'r2':
           return vector * self.coors ** 2
        elif factor == 'linear':
           return vector
        elif factor == 'spherical' or True:
           return vector * self.coors ** 2 * 4 * math.pi
        raise Exception('Unknown integral factor: %s' % factor)

    def scale(self, scale=None, shift = None):
        if scale:
           self.coors *=scale
        if shift:
           self.coors +=shift

    def linear_second_derivation(self, vector,  from_zero = False, at = None):
        """
        Return vector from second derivations of given vector
        .. math::
        """
        return self.second_derivation(self, vector, 'linear', from_zero, at)

    def second_derivation(self, vector, factor = None, from_zero = False, at = None):
        """
        Return vector from second derivations of given vector that was integrated using given integral factor
        .. math::
        """
        vector = self.mul_integral_factor(vector, factor)

        spaces = np.convolve(self.coors, [1, -1])[1:-1]
        diffs = np.convolve(vector, [1, -1])[1:-1]

        out = RadialVector(self, np.zeros(self.size))
        #             f1       * (x2 / x1) +                      f2
        out.values[1:-1] =  diffs[:-1]  * (spaces[1:] / spaces[:-1]) - diffs[1:]
        out.values[1:-1] /= spaces[1:]*spaces[1:] + spaces[1:]*spaces[:-1]
        out.values[1:-1] *= -2
        #border
        if(from_zero):
          out.values[0] =  vector[0] * spaces[0] / self.coors[0] - diffs[0]
          out.values[0] /= -2*spaces[0] * spaces[0] + spaces[0] * self.coors[0]
        else:
          out.values[0]= (diffs[1] / spaces[1] - diffs[0] / spaces[0] ) / (spaces[1] )
        out.values[-1]= -(diffs[-2] / spaces[-2] - diffs[-1] / spaces[-1] ) / (spaces[-1] )

        if at:
           return out.interpolate(at)
        return out

    """
    Return radial function, that 'dimension'-D spherical integral is given as 'vector'.
    2D
    .. math ::
       g_n = f_n' / (2 * pi * r)
    .. math ::
       g_n = f_n' / (4 * pi * r**2)
    """
    def space_derivation(self, vector, dimension = 3, from_zero = False, at = None):
        out = self.derivation(vector, 'linear', from_zero, at)
        if dimension == 3:
           out = out / (4 * math.pi) / self.coors / self.coors
        elif dimension == 2:
           out = out / (2 * math.pi) / self.coors
        else:
           raise Exception('Please implement {}-dimensional sphere surface in radial_mesh.RadialMesh.space_derivation'.format(dimension))
        if self.coors[0] == 0.0:
           f=lambda x,a,b,c,d,e: ((((a*x)+b)*x+c)*x)+d
           out[0]=f(0.0,*curve_fit(f, self.coors[1:15], out[1:15])[0])
        return out

    def derivation(self, vector, factor = None, from_zero = False, at = None):
        """
        Return vector from derivations of given vector that was integrated using given integral factor
        .. math::
          f_n = \frac{v(x_n - \epsilon) - v(x_n + \epsilon)}{2 * \epsilon * factor}
        """
        vector = self.mul_integral_factor(vector, factor)
        spaces = np.convolve(self.coors, [1, -1])[1:-1]
        diffs = np.convolve(vector, [1, -1])[1:-1]
        diffs /= spaces
        out = np.convolve(diffs, [0.5, 0.5])
        if(from_zero):
          out[0] = (diffs[0] + vector[0]/spaces[0]) / 2
        else:
          out[0] = diffs[0]
        out[-1] = diffs[-1]
        out = RadialVector(self, out)
        if at:
           return out.interpolate(at)
        return out

    def linear_derivation(self, vector, at = None, from_zero = False):
        """
        Return vector from derivations of given vector (with no integral factor)
        """
        return self.derivation(vector, factor = 'linear',  from_zero = False, at = at)

    def interpolate_3d(self, potential, coors, centre=None, kind=None, out=None):
        """
        Interpolate values in points giving by 3d coordinates with respect to given centre.
        """
        if centre is None:
           centre = np.zeros((3,))
        return self.interpolate(potential, coors, centre, kind, out)

    def integrate(self, vector, factor = None):
        """
        .. math::
           \int f(r) r^2 dr
        """
        v = self.mul_integral_factor(vector, factor)
        return simps(v, self.coors)

    def linear_integrate(self, vector):
        """
        .. math::
           \int f(r) dr
        """
        return simps(vector, self.coors)

    def get_coors(self):
        return self.coors

    def integral(self, vector, factor = None, from_zero=False):
        """
        .. math::
          a_n = \int_{r_0}^{r_n} f(r) * integral_factor dr

        from_zero starts to integrate from zero, instead of starting between
        the first two points
        """
        v = self.mul_integral_factor(vector, factor)
        r = self.coors
        if from_zero:
            v = cumtrapz(vector, r, initial=0) + v[0] / 2 * max(0.0, r[0])
            return RadialVector(self, v)
        else:
            v = cumtrapz(vector, r)
            r = r[1:]
            return RadialVector(r, v)

    def linear_integral(self, vector, from_zero=False):
        """
        .. math::
          a_n = \int_{r_0}^{r_n} f(r) dr

        from_zero starts to integrate from zero, instead of starting between
        the first two points
        """
        return self.integral(vector, 'linear', from_zero)


    def dot(self, vector_a, vector_b, factor= None):
        """
        Dot product with respect to given integral factor (see RadialMesh.integral_factor)
        .. math::
           \int f(r) g(r) factor r^2 dr
        """
        vector_a=self._get_values_from_object(vector_a)
        vector_b=self._get_values_from_object(vector_b)
        return self.integrate(vector_a * vector_b, factor)

    def norm(self, vector, factor = None):
        """
        L2 v norm with respect to given integral factor (see RadialMesh.integral_factor)

        .. math::
           \sqrt(\int f(r) f(r) factor r^2 dr)
        """
        return np.sqrt(self.dot(vector, vector, factor))

    def self_scalar_product(self, vec_a, vec_b):
        return self.dot(self * vec_a, vec_b)


    def output_vector(self, vector, filename=None, other = None):
        """
        Output vector or vectors in columns prepended by mesh coordinates to file or stdout if no filename given.
        Output format
        corrs[0] v[0]
        corrs[1] v[1]
        etc... or
        corrs[0] v0[0] v1[0]] ...
        corrs[1] v0[1] v1[1]] ...
        """
        if filename is None:
            import sys
            filename = sys.stdout
        if isinstance(vector, (RadialVector, np.ndarray)):
            vector = [vector]

        if isinstance(other, (RadialVector, np.ndarray)):
           vector.append(other)
        elif(other is not None):
           vector.extend(other)

        if isinstance(vector, (list, tuple)):
          vector = [ v.values if isinstance(v, RadialVector) else v for v in vector ]
          vector = np.vstack([self.coors] + vector)
        np.savetxt(filename, vector.T)

    def plot(self, vector, cmd='plot', range=None, other = None):
        """ Plot given vectors using given shell script"""
        import tempfile
        import os
        if range:
           if isinstance(range, (tuple, list)):
              range = '%f:%f' % range
           else:
              range.replace(",",":")
           range='-r' + range
        else:
           range=''
        fhandle, fname = tempfile.mkstemp()
        fil = os.fdopen(fhandle, 'w')
        self.output_vector(vector, fil, other = other)
        fil.close()
        os.system('%s %s %s' % (cmd, fname, range))
        os.remove(fname)

    @staticmethod
    def merge(meshes):
        """ Merge more radial meshes to one """
        merged = np.concatenate(tuple(m.coors for m in meshes))
        return RadialMesh(np.unique(merged))

    def intervals(self):
        """ Return distances between coors """
        return np.convolve(self.coors, [1, -1])[1:-1]

    def __eq__(self, mesh):
        if mesh is self: return True
        if isinstance(mesh, np.ndarray):
           return (mesh == self.coors).all()
        if isinstance(mesh, RadialMesh):
           return (mesh.coors == self.coors).all()
        return

    def __getitem__(self, i):
        return self.coors[i]


    def orthogonalize(self, vectors, scalar_product = None, b_operator=None, factor=None, reorthogonalization = 0):
      """ Orthogonalization of given vectors.

          Scalar product can be function of two numpy array, or None (then standard scalar
          product with given integral factor factor is used)
          If b_operator is not None, then b_operator-scalar product is used and so B-orthogonalization
          is performed. Operator can be given as radial vector representing local operator.
      """

      if not isinstance(vectors, np.ndarray):
          vectors = [ self._get_values_from_object(v) for v in vectors ]
      if isinstance(scalar_product, RadialVector):
         b_operator = scalar_product
         scalar_product = None
      if scalar_product is None:
         scalar_product = lambda x,y: self.dot(x,y,factor)
      if isinstance(b_operator, RadialVector):
         proj_vector = b_operator
         b_operator = lambda x: x * proj_vector

      orto, c_coef, vorto = orthogonalize(vectors, scalar_product, b_operator, reorthogonalization)
      r = len(orto)
      vec = range(r)
      vvec = range(r)
      for i in xrange(r):
        vec[i] = RadialVector(self, orto[i])
        vvec[i] = RadialVector(self, vorto[i]) if b_operator else vec[i]
      return vec, c_coef, vvec




class RadialMesh(BaseRadialMesh):
    """   Radial mesh given by explicit mesh point coordinates   """
    def __init__(self, coors, factor = 'spherical'):
        self.coors = np.asarray(coors)
        self.midpointMesh = {}
        self.parentMesh = None
        self.integral_factor = factor

    def copy(self):
        out = copy.copy(self)
        out.coors= self.coors.copy()
        return out

    @property
    def shape(self):
        return self.coors.shape

    @property
    def size(self):
        return self.coors.size

    def get_index_r(self, value):
        if isinstance(value, int):
           return value, self.get_r(value)
        return self.get_index(value), value

    def get_coors(self):
        return self.coors

    def last_point(self):
        return self.coors[self.coors.size - 1]

    def get_r(self, index):
        return self.coors[index]

    def get_index(self, r):
        pos = self.coors.searchsorted(r)
        return (pos if pos < self.coors.size else self.coors.size - 1)

    def get_mixing(self, r):
        pos = self.get_index(r)
        if pos == self.coors.size - 1 and self.coors[pos] < r:
            out = [(pos, 1.0)]

        elif pos == 0 or self.coors[pos] == r:
            out = [(pos, 1.0)]

        else:
            pos_c = (r - self.coors[pos - 1]) / (self.coors[pos]
                                                 - self.coors[pos - 1])
            out = [(pos - 1, 1.0 - pos_c), (pos, pos_c)]

        return out

    def interpolator(self, potential, kind = 'cubic'):
        return si.interp1d(self.coors, potential, kind, copy = False, bounds_error = False, fill_value = potential[-1])

    def interpolate(self, potential, r , centre=None, kind = None, out=None):
        if centre is not None:
           if centre.any():
              r = r - centre
           r = norm_l2_along_axis(r, axis=1)
        if isinstance(potential, RadialVector):
           potential = potential.values
        if kind is None:
           val = np.interp(r, self.coors, potential, right=0)
        else:
           val = self.interpolator(potential, kind)(r)
        if out:
           out[:]=val
           return out
        return val
        #return np.interp(r, self.coors, potential, right=0)

    def get_midpoint_mesh(self, to=None):
        if to is None:
            to = len(self.coors)
        else:
            if not isinstance(to, int):
                to = self.get_r(to)
        if self.midpointMesh.has_key(to):
            return self.midpointMesh[to]

        if to is None:
            coors = self.coors
        else:
            coors = self.coors[0:to]

        midpoints = np.convolve(coors, [0.5, 0.5], 'valid')
        midmesh = RadialMesh(midpoints)
        self.midpointMesh[to] = midmesh
        midmesh.parentMesh = self
        return midmesh

    def get_parent_mesh(self):
        return self.parentMesh

    def slice(self, x, y):
        if isinstance(x, float):
            x = self.get_index(x)
        if isinstance(y, float):
            y = self.get_index(y)
        return RadialMesh(self.coors[x:y])

    def sparse_vector(self, vector):
        values = np.tile(float('NAN'), self.size)
        ii = self.coors.searchsorted(vector.mesh.coors)
        values[ii] = vector.values
        return RadialVector(self, values)

    def partition(self):
        at = np.where(self.coors[1:]<self.coors[:-1])[0]+1
        return [RadialMesh(v) for v in np.split(self.coors, at)]





class BaseParametricMesh(RadialMesh):
    def __init__(self, params, coors, factor):
        self.params = params
        super(BaseParametricMesh, self).__init__(coors, factor)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.params == other.params;

    def __ne__(self, other):
        return not isinstance(other, self.__class__) or self.params != other.params;

    def __new__(self, *params):
        """ Parametric meshes are reused so only one is created. Mesh is immutable class."""
        if not hasattr(self, 'instance_cache'):
           self.instance_cache = {}
        hsh = hash(params)
        if not self.instance_cache.has_key(hsh):
           self.instance_cache[hsh] = super(BaseParametricMesh, self).__new__(self, *params)
        return self.instance_cache[hsh]

class HyperbolicMesh(BaseParametricMesh):
    """   Radial hyperbolic mesh given by params, see __init__"""

    def __init__(self, jm, ap=None, size=None, from_zero=False, factor='spherical'):
        """
        If three (jp, ap, size) params is given, mesh has size points at
         a(n) = ap * n  / ( jm - n)
        If two params given, mesh has ap points hyperbolicaly distributed over interval
           (0 if from_zero else 1, jm)
        If one params given, mesh has jm points hyperbolicaly distributed over interval
           (0 if from_zero else 1, jm)
        """
        if size is None:
            # range, number of points
            self.size_par = (ap if not ap is None else jm)
            self.ap = 1.0
            self.jm = self.size_par / jm + self.size_par
        else:
            # clasical
            self.size_par = size
            self.jm = jm
            self.ap = ap

        coors = np.arange((0.0 if from_zero else 1.0), self.size_par + 1,
                          dtype=np.float64)
        coors = self.ap * coors / (self.jm - coors)

        super(HyperbolicMesh, self).__init__((self.jm, self.ap, self.size_par, from_zero), np.asfortranarray(coors), factor)

class LogspaceMesh(BaseParametricMesh):
      def __init__(self, start, stop, num=50, endpoint = True, base = 10, factor='spherical'):
          super(LogspaceMesh, self).__init__((start, stop, num, endpoint, base), np.logspace(start, stop, num, endpoint, base), factor)

      @staticmethod
      def create(from_val, to_val, num):
          """
              all float:            (start, stop, exponent)
              from_val is integer:  (start, num values, step)
              num is integer:       (start, stop, num values)
          """

          if isinstance(num, int):
             args = (math.log(from_val), math.log(to_val), num, True, math.e)
          elif isinstance(to_val, int):
             frm = math.log(from_val)
             args = (frm, frm + math.log(num)*to_val , to_val, False, math.e)
          else:
             lg = math.log(num)
             count = math.log(to_val / from_val) / lg
             args =  (math.log(from_val) / lg, math.log(to_val) / lg, count, True, num)
          return LogspaceMesh(*args)

class EquidistantMesh(BaseParametricMesh):
    def __init__(self, num, from_val=0.0, to_val=1.0, factor = 'spherical'):
        if isinstance(num, float):
           num = int((to_val - from_val) / num + 1)
        distance = (to_val - from_val) / (num - 1)
        super(EquidistantMesh, self).__init__((from_val, to_val, num), np.arange(num) * distance + from_val, factor)

