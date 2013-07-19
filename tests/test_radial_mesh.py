from sfepy.base.testing import TestCommon
import numpy as np
from sfepy.physics.radial_mesh import RadialVector as RV, ExplicitRadialMesh as RM
import inspect

def equals(f, msg):
    e=None
    try:
      if f():
        return True
    except Exception, ee:
      e=ee 


    frame,filename,line_number,function_name,lines,index=\
       inspect.getouterframes(inspect.currentframe())[1]
    msg = "Error at %s at line %i with msg: %s" %(filename,line_number, msg)
    if e:
       print msg, " - exception ", e
       import traceback
       traceback.print_exc()
    else:
       print msg, " - failure"
    raise Exception(msg)
    
##
# 02.07.2007, c
class Test( TestCommon ):

    @staticmethod
    def from_conf( conf, options ):
        return Test( conf = conf, options = options )

    def test_derivation(self):
        m = RM([1,2,3])
        v = RV(m,[1,4,9])
        equals(lambda: v.linear_derivation() == [3, 4, 5], "test derivation")
        equals(lambda: v.linear_integral() == [2.5, 9], "test integral")
        equals(lambda: v.linear_integral(from_zero = True) == [0.5, 3, 9.5], "integral from zero")
        return True



