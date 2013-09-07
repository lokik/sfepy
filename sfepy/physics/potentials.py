"""
Classes for constructing potentials of atoms and molecules.
"""

import numpy as nm
from sfepy.physics.radial_mesh import RadialVector, BaseRadialMesh
from sfepy.base.base import as_float_or_complex, Container, Struct
from sfepy.linalg import norm_l2_along_axis

class CompoundPotential(Container):
    """
    Sum of several potentials.
    """

    def __init__(self, objs=None):
        Container.__init__(self, objs=objs)

    def insert(self, ii, obj):
        Container.insert(self, ii, obj)

    def append(self, obj):
        Container.append(self, obj)

    def __mul__(self, other):
        out = CompoundPotential()
        for name, pot in self.iteritems():
            out.append(pot * other)

        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, PotentialBase):
            out = self.copy()
            out.append(other)

        elif isinstance(other, CompoundPotential):
            out = self.__class__(self._objs + other._objs)

        else:
            raise ValueError('cannot add CompoundPotential with %s!' % other)

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, PotentialBase):
            out = self + (-other)

        elif isinstance(other, CompoundPotential):
            out = self + (-other)

        else:
            raise ValueError('cannot subtract CompoundPotential with %s!' \
                             % other)

        return out

    def __rsub__(self, other):
        return -self + other

    def __pos__(self):
        return self

    def __neg__(self):
        return -1.0 * self

    def __call__(self, coors):
        val = 0.0
        for pot in self:
            val += pot(coors)

        return val

class PotentialBase(Struct):
    """
    Base class for potentials.
    """
    compoud_potential_class = CompoundPotential

    def __mul__(self, other):
        try:
            mul = as_float_or_complex(other)

        except ValueError:
            raise ValueError('cannot multiply PotentialBase with %s!' % other)

        out = self.copy(name=self.name)

        out.sign = mul * self.sign

        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, PotentialBase):
            out = self.compoud_potential_class([self, other])

        elif nm.isscalar(other):
            if other == 0:
                out = self

            else:
                out = NotImplemented

        else:
            out = NotImplemented

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self - 1.0 * other

    def __rsub__(self, other):
        return -self + other

    def __pos__(self):
        return self

    def __neg__(self):
        out = -1.0 * self
        return out


class BaseSphericalPotential(PotentialBase):
    """
    Single spherically symmetric potential.
    """

    def __init__(self, name, centre=None, dim=3):
        self.name = name
        if centre is None:
            self.centre = None # nm.array([0.0] * dim, dtype=nm.float64)
        else:
            self.centre = nm.asarray(centre, dtype=nm.float64)
        self.sign = 1.0

    def __call__(self, coors):
        coors = self.get_distance(coors)
        pot = self.function(coors)
        if self.sign != 1.0:
           pot *= self.sign 
        return pot

    def __iter__( self ):
        """
        Allow iteration even over a single potential.
        """
        yield self

    def __len__(self):
        """
        Allow length even of a single potential.
        """
        return 1

    def get_centre(self):
        """ 
        Return potential centre, for redefining in descendants 
        """
        if hasattr(self.centre, 'get_centre'):
           return self.centre.get_centre()
        return self.centre
           

    def get_distance(self, coors):
        """
        Get the distance of points with coordinates `coors` of the
        potential centre.
        """
        if isinstance(coors, BaseRadialMesh):
           return coors.coors
        if isinstance(coors, (int, float)):
           return float(coors)      
        if len(coors.shape) == 1:
           return coors           
        if self.centre is None:
          return norm_l2_along_axis(coors)  
        if(hasattr(self.centre, 'get_distance')):
           return self.centre.get_distance(coors)
        return norm_l2_along_axis(coors - self.get_centre())

    def get_derivative(self, coors, eps=1e-6):
        """ 
        Return its derivative in given coordinates. In this case is numerically computed
        using \epsilon = eps
        In case the function is Potential, let the derivation computation on the Potential itself.
        (so the continuity of potential derivations is retained)
        """
        if hasattr(self.function, 'get_derivative'):
           return self.function.get_derivative(coors, eps)
        r = self.get_distance(coors)
        fp1 = self(r + eps)
        fm1 = self(r - eps)
        d1 = (fp1 - fm1) / (2.0 * eps)
        return d1

    def get_second_derivative(self, coors, eps=1e-6):
        """ Return its second derivative in given coordinates. In this case is numerically 
        computed using \epsilon = eps
        In case the function is Potential, let the derivation computation on the Potential itself.
        (so the continuity of potential derivations is retained)
        """
        r = self.get_distance(coors)
        if hasattr(self.function, 'get_second_derivative'):
           return self.function.get_second_derivative(r, eps)
           
        f0 = self(r)
        fp2 = self(r + 2.0 * eps)
        fm2 = self(r - 2.0 * eps)
        # Second derivative w.r.t. r.
        d2 = (fp2 - 2.0 * f0 + fm2) / (4.0 * eps * eps)
        return d2        

    def get_charge(self, coors, eps=1e-6, brute_force =False):
        """
        Get charge corresponding to the potential by numerically
        applying Laplacian in spherical coordinates.
        """
        r = self.get_distance(coors)
        d1 = self.get_derivative(r)
        d2 = self.get_second_derivative(r)
        # First derivative w.r.t. r.
        charge = - self.sign / (4.0 * nm.pi) * (d2 + 2.0 * d1 / r)
        return charge


class SphericalPotential(BaseSphericalPotential):
    """ spherically symmetric potential given by radial function """
    def __init__(self, name, function, centre=None, dim=3, args=None):
        if args:
           self.function = lambda x: function(x, *args)
        else:
           self.function = function
        super(SphericalPotential, self).__init__(name, centre, dim)

    def get_charge(self, coors, eps=1e-6, brute_force =False):
        """
        Get charge corresponding to the potential by numerically
        applying Laplacian in spherical coordinates.
        """
        
        r = self.get_distance(coors)
        if hasattr(self.function, 'get_charge') and not brute_force:
           return self.function.get_charge(r, eps)
        BaseSphericalPotential.get_charge(self, r, eps)


class DiscreteSphericalPotential(BaseSphericalPotential):
    """ spherically symmetric potential given by discrete values """
    def __init__(self, name, mesh = None, values=None, centre=None, dim=3):
        if values is None:
           self.vector = mesh
        else:
           self.vector = RadialVector(mesh, values)
        super(DiscreteSphericalPotential, self).__init__(name, centre, dim)

    @property
    def mesh(self):
         return self.vector.mesh

    @property
    def values(self):
         return self.vector.values

    def function(self, r):
        return self.vector.interpolate(r, centre = self.centre)
        
    def get_derivative(self, r = None, eps = None):
        if r is not None:
           r = self.get_distance(r)
        return self.vector.linear_derivation(at = r)

    def get_second_derivative(self, r = None, eps = None):
        if r is not None:
           r = self.get_distance(r)
        return self.vector.linear_second_derivation(at = r) 

    def get_charge(self, coors, eps=1e-6):
        """
        Get charge corresponding to the potential by numerically
        applying Laplacian in spherical coordinates.
        """
        d1 = self.vector.linear_derivation()
        d2 = self.vector.linear_second_derivation()
        mcoors =  d1.mesh.coors
        charge = - self.sign / (4.0 * nm.pi) * (d2 + 2.0 * d1 / mcoors)
        return charge.interpolate(self.get_distance(coors))        
        
        """ another formula, the first is more precise """
        r2 = mcoors**2
        bracket = d1 * r2                
        r = self.get_distance(coors)
        return - self.sign * (bracket.linear_derivation() / r2 / 4 * nm.pi).interpolate(r) 

    def __add__(self, other):
        if isinstance(other, DiscreteSphericalPotential):
            out = self.copy(name = self.name)
            out.vector = self.vector * self.sign + other.vector * other.sign
            out.sign = 1.0
        else:
            out = super(DiscreteSphericalPotential, self).__add__(other)
        return out
