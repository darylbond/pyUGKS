## \file cns_bc_defs.py
## \ingroup mb_cns
##
## \author P.Jacobs
## \version 31-Jan-2005 extracted from e_model_spec.py
##
"""
Dictionary to look up boundary-condition index from name or number.

Boundary conditions are implemented within the simulation by setting
flow data in ghost cells to suitable values.
This is done once per time step, before evaluating the fluxes.

@var ADJACENT: This boundary joins that of another block.
    Normally, this boundary condition would be set implicitly
    when making block connections.
@type ADJACENT: int
@var COMMON: Synonym for L{ADJACENT}.
@var EXTRAPOLATE_OUT: Extrapolate all flow properties from
   just inside the boundary into the ghost-cells outside
   the boundary.  This works fine for a strong supersonic outflow.
"""

ADJACENT        = 0
COMMON          = 0
EXTRAPOLATE_OUT = 1
CONSTANT        = 2
REFLECT         = 3
ACCOMMODATING   = 4
PERIODIC        = 5

bcIndexFromName = {
     0: ADJACENT, "0": ADJACENT, "ADJACENT": ADJACENT, "COMMON": ADJACENT,
     1: EXTRAPOLATE_OUT, "1":  EXTRAPOLATE_OUT, "EXTRAPOLATE_OUT": EXTRAPOLATE_OUT,
     2: CONSTANT, "2": CONSTANT,
     3: REFLECT, "3": REFLECT,
     4: ACCOMMODATING, "4": ACCOMMODATING,
     5: PERIODIC, "5":PERIODIC,
}

bcName = {
    ADJACENT: "ADJACENT",
    EXTRAPOLATE_OUT: "EXTRAPOLATE_OUT",
    CONSTANT: "CONSTANT",
    REFLECT: "REFLECT",
    ACCOMMODATING: "ACCOMMODATING",
    PERIODIC: "PERIODIC",
    }

class BoundaryCondition(object):
    """
    Base class for boundary condition specifications.

    type_of_BC: specifies the boundary condition
    T: fixed wall temperature (in degrees K) that will be used if
        the boundary conditions needs such a value.
    """
    __slots__ = 'type_of_BC', 'D', 'T', "U", "V", 'other_block',\
            'other_face', 'orientation', 'label', 'UDF_U',\
            'UDF_V', 'UDF_T','UDF_D'
            
    def __init__(self,
                 type_of_BC = REFLECT,
                 D = 1.0,
                 T = 300.0,
                 U = 0.0,
                 V = 0.0,
                 UDF_D = None,
                 UDF_U = None,
                 UDF_V = None,
                 UDF_T = None,
                 other_block=-1,
                 other_face=-1,
                 orientation=0,
                 label=""):
                     
        self.type_of_BC = type_of_BC
        self.D = D
        self.T = T
        self.U = U
        self.V = V
        self.UDF_D = UDF_D
        self.UDF_U = UDF_U
        self.UDF_V = UDF_V
        self.UDF_T = UDF_T
        self.other_block = other_block
        self.other_face = other_face
        self.orientation = orientation
        self.label = label
            
        return
    def __str__(self):
        str_rep = "BoundaryCondition("
        str_rep += "type_of_BC=%d" % self.type_of_BC
        str_rep += ", D=%g" % self.D
        str_rep += ", T=%g" % self.T
        str_rep += ", U=%g" % self.U
        str_rep += ", V=%g" % self.V
        str_rep += ", UDF_D=%s" % self.UDF_D
        str_rep += ", UDF_U=%s" % self.UDF_U
        str_rep += ", UDF_V=%s" % self.UDF_V
        str_rep += ", UDF_T=%s" % self.UDF_T
        str_rep += ", other_block=%d" % self.other_block
        str_rep += ", other_face=%d" % self.other_face
        str_rep += ", orientation=%d" % self.orientation
        str_rep += ", label=\"%s\")" % self.label
        return str_rep
    def __copy__(self):
        return BoundaryCondition(type_of_BC=self.type_of_BC,
                                 D=self.D,
                                 T=self.T,
                                 U=self.U,
                                 V=self.V,
                                 UDF_D = self.UDF_D,
                                 UDF_U = self.UDF_U,
                                 UDF_V = self.UDF_V,
                                 UDF_T = self.UDF_T,
                                 other_block=self.other_block,
                                 other_face=self.other_face,
                                 orientation=self.orientation,
                                 label=self.label)
    
class AdjacentBC(BoundaryCondition):
    """
    This boundary joins (i.e. is adjacent to) a boundary of another block.

    This condition is usually not set manually but is set as part of the
    connect_blocks() function.
    """
    def __init__(self, other_block=-1, other_face=-1, orientation=0, label="ADJACENT"):
        BoundaryCondition.__init__(self, type_of_BC=ADJACENT, other_block=other_block,
                                   other_face=other_face, orientation=orientation,
                                   label=label)
        return
    def __str__(self):
        return "AdjacentBC(other_block=%d, other_face=%d, orientation=%d, label=\"%s\")" % \
            (self.other_block, self.other_face, self.orientation, self.label)
    def __copy__(self):
        return AdjacentBC(other_block=self.other_block,
                          other_face=self.other_face,
                          orientation=self.orientation,
                          label=self.label)
class PeriodicBC(BoundaryCondition):
    """
    This boundary joins (i.e. is adjacent to) a boundary of another block.
    """
    def __init__(self, other_block=-1, other_face=-1, orientation=0, label="PERIODIC"):
        BoundaryCondition.__init__(self, type_of_BC=ADJACENT, other_block=other_block,
                                   other_face=other_face, orientation=orientation,
                                   label=label)
        return
    def __str__(self):
        return "PeriodicBC(other_block=%d, other_face=%d, orientation=%d, label=\"%s\")" % \
            (self.other_block, self.other_face, self.orientation, self.label)
    def __copy__(self):
        return PeriodicBC(other_block=self.other_block,
                          other_face=self.other_face,
                          orientation=self.orientation,
                          label=self.label)

class ExtrapolateOutBC(BoundaryCondition):
    """
    Fill the ghost cells with data from just inside the boundary.

    This boundary condition will work best if the flow is supersonic,
    directed out of the flow domain.
    """
    def __init__(self, label="EXTRAPOLATE OUT"):
        BoundaryCondition.__init__(self, type_of_BC=EXTRAPOLATE_OUT,
                                   label=label)
        return
    def __str__(self):
        return "ExtrapolateOutBC(label=\"%s\")" % self.label
    def __copy__(self):
        return ExtrapolateOutBC(label=self.label)

class ConstantBC(BoundaryCondition):
    """
    fill the ghost cells with data that is constant throughout the simulation
    run. This data is loaded at the initialisation stage and then left untouched
    """
    def __init__(self, D=0, T=0, U=0, V=0, UDF_D=None,
                 UDF_U=None, UDF_V=None, UDF_T=None, label="CONSTANT"):
        BoundaryCondition.__init__(self, U=U, V=V, T=T, 
                                   UDF_D=UDF_D, UDF_U=UDF_U, UDF_V=UDF_V, UDF_T=UDF_T, 
                                   type_of_BC=CONSTANT, label=label)
        return
    def __str__(self):
        return "ConstantBC(D=%d, T=%d, U=%d, V=%d, UDF_D=%s, UDF_U=%s, UDF_V=%s, UDF_T=%s, label=\"%s\")" % \
            (self.D, self.T, self.U, self.V, self.UDF_D, 
             self.UDF_D, self.UDF_U, self.UDF_V, self.UDF_T, self.label)
    def __copy__(self):
        return ConstantBC(D=self.D, T = self.T,
                          U = self.U, V = self.V,
                          UDF_D=self.UDF_D, UDF_U=self.UDF_U, 
                          UDF_V=self.UDF_V, UDF_T=self.UDF_T,
                          label=self.label)
    
class ReflectBC(BoundaryCondition):
    """
    fill the ghost cells with data that is a reflection of the data in the flow 
    domain about the plane of the boundary
    """
    def __init__(self, label="REFLECT"):
        BoundaryCondition.__init__(self, type_of_BC=REFLECT, label=label)
        return
        
    def __str__(self):
        return "ReflectBC(label=\"%s\")" % self.label
    def __copy__(self):
        return ReflectBC(label=self.label)

class AccommodatingBC(BoundaryCondition):
    """
    define the values used to define the equilibrium distribution for the wall
    U = velocity of wall, parallel to wall, positive in direction of cell wall tangent vector
    """
    def __init__(self, D=1, T=0, U=0, V=0, UDF_D=None, 
                 UDF_U=None, UDF_V=None, UDF_T=None, label="ACCOMMODATING"):
        BoundaryCondition.__init__(self, D=D, U=U, 
                                   V=V, T=T, UDF_D=UDF_D, 
                                   UDF_U=UDF_U, UDF_V=UDF_V, UDF_T=UDF_T, 
                                   type_of_BC=ACCOMMODATING, label=label)
        
        return
    def __str__(self):
        return "AccommodatingBC(D=%g, T=%g, U=%g, V=%g, UDF_D=%s, UDF_U=%s, UDF_V=%s, UDF_T=%s, label=\"%s\")" % \
            (self.D, self.T, self.U, self.V, self.UDF_D, self.UDF_U, self.UDF_V, self.UDF_T, self.label)
    def __copy__(self):
        return AccommodatingBC(D=self.D,
                               T = self.T,
                               U = self.U, 
                               V = self.V,
                               UDF_D = self.UDF_D,
                               UDF_U = self.UDF_U, 
                               UDF_V = self.UDF_V,
                               UDF_T = self.UDF_T,
                               label=self.label)