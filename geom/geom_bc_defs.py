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
DIFFUSE         = 4
PERIODIC        = 5

bcIndexFromName = {
     0: ADJACENT, "0": ADJACENT, "ADJACENT": ADJACENT, "COMMON": ADJACENT,
     1: EXTRAPOLATE_OUT, "1":  EXTRAPOLATE_OUT, "EXTRAPOLATE_OUT": EXTRAPOLATE_OUT,
     2: CONSTANT, "2": CONSTANT,
     3: REFLECT, "3": REFLECT,
     4: DIFFUSE, "4": DIFFUSE,
     5: PERIODIC, "5":PERIODIC,
}

bcName = {
    ADJACENT: "ADJACENT",
    EXTRAPOLATE_OUT: "EXTRAPOLATE_OUT",
    CONSTANT: "CONSTANT",
    REFLECT: "REFLECT",
    DIFFUSE: "DIFFUSE",
    PERIODIC: "PERIODIC",
    }

class BoundaryCondition(object):
    """
    Base class for boundary condition specifications.

    type_of_BC: specifies the boundary condition
    Twall: fixed wall temperature (in degrees K) that will be used if
        the boundary conditions needs such a value.
    """
    __slots__ = 'type_of_BC', 'Dwall', 'Twall', "Uwall", "Vwall", 'other_block',\
            'other_face', 'orientation', 'label'
            
    def __init__(self,
                 type_of_BC = REFLECT,
                 Dwall = 1.0,
                 Twall = 300.0,
                 Uwall = 0.0,
                 Vwall = 0.0,
                 other_block=-1,
                 other_face=-1,
                 orientation=0,
                 label=""):
                     
        self.type_of_BC = type_of_BC
        self.Dwall = Dwall
        self.Twall = Twall
        self.Uwall = Uwall
        self.Vwall = Vwall
        self.other_block = other_block
        self.other_face = other_face
        self.orientation = orientation
        self.label = label
            
        return
    def __str__(self):
        str_rep = "BoundaryCondition("
        str_rep += "type_of_BC=%d" % self.type_of_BC
        str_rep += ", Dwall=%g" % self.Dwall
        str_rep += ", Twall=%g" % self.Twall
        str_rep += ", Uwall=%g" % self.Uwall
        str_rep += ", other_block=%d" % self.other_block
        str_rep += ", other_face=%d" % self.other_face
        str_rep += ", orientation=%d" % self.orientation
        str_rep += ", label=\"%s\")" % self.label
        return str_rep
    def __copy__(self):
        return BoundaryCondition(type_of_BC=self.type_of_BC,
                                 Dwall=self.Dwall,
                                 Twall=self.Twall,
                                 Uwall=self.Uwall,
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
        BoundaryCondition.__init__(self, type_of_BC=PERIODIC, other_block=other_block,
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
    def __init__(self, Dwall = 1.0, Twall = 300.0, Uwall = 0.0, Vwall = 0.0, 
                 label="CONSTANT"):
        BoundaryCondition.__init__(self, type_of_BC=CONSTANT, Dwall = Dwall, 
                                   Twall = Twall, Uwall = Uwall, Vwall = Vwall, 
                                   label=label)
        return
    def __str__(self):
        return "ConstantBC(Dwall=%d, Twall=%d, Uwall=%d, Vwall=%d label=\"%s\")" % \
            (self.Dwall, self.Twall, self.Uwall, self.Vwall, self.label)
    def __copy__(self):
        return ConstantBC(Dwall=self.Dwall, Twall = self.Twall,
                          Uwall = self.Uwall, Vwall = self.Vwall, label=self.label)
    
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

class DiffuseBC(BoundaryCondition):
    """
    define the values used to define the equilibrium distribution for the wall
    Uwall = velocity of wall, parallel to wall, positive in direction of cell wall tangent vector
    """
    def __init__(self, Twall = 300.0, Uwall = 0.0, label="DIFFUSE"):
        BoundaryCondition.__init__(self, type_of_BC=DIFFUSE, Twall = Twall, 
                                   Uwall = Uwall, label=label)
        return
    def __str__(self):
        return "DiffuseBC(Twall=%d, Uwall=%d, label=\"%s\")" % \
            (self.Twall, self.Uwall, self.label)
    def __copy__(self):
        return DiffuseBC(Twall = self.Twall,
                          Uwall = self.Uwall, label=self.label)