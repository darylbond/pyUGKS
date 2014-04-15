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
from geom_flow import FlowCondition
import numpy as np
import copy

ADJACENT        = 0
COMMON          = 0
EXTRAPOLATE_OUT = 1
CONSTANT        = 2
REFLECT         = 3
DIFFUSE         = 4
PERIODIC        = 5
INFLOW          = 6
OUTFLOW         = 7
ADSORBING       = 8
BOUNCE_BACK     = 9

# flip distribution functions
NO_FLIP = 0
FLIP_NS = 1
FLIP_D  = 2
FLIP_EW = 3

# block relative orientation
NO_TRANSFORM = 0
TRANSFORM = 1
MIRROR = 2

HOLD = 0  # let the code do what it wants
NO_HOLD = 1 # do not alter anything!




bcIndexFromName = {
     0: ADJACENT, "0": ADJACENT, "ADJACENT": ADJACENT, "COMMON": ADJACENT,
     1: EXTRAPOLATE_OUT, "1":  EXTRAPOLATE_OUT, "EXTRAPOLATE_OUT": EXTRAPOLATE_OUT,
     2: CONSTANT, "2": CONSTANT,
     3: REFLECT, "3": REFLECT,
     4: DIFFUSE, "4": DIFFUSE,
     5: PERIODIC, "5":PERIODIC,
     6: INFLOW, "6":INFLOW,
     7: OUTFLOW, "7":OUTFLOW,
     8: ADSORBING, "8":ADSORBING,
     9: BOUNCE_BACK, "9":BOUNCE_BACK
}

bcName = {
    ADJACENT: "ADJACENT",
    EXTRAPOLATE_OUT: "EXTRAPOLATE_OUT",
    CONSTANT: "CONSTANT",
    REFLECT: "REFLECT",
    DIFFUSE: "DIFFUSE",
    PERIODIC: "PERIODIC",
    INFLOW: "INFLOW",
    OUTFLOW: "OUTFLOW",
    ADSORBING: "ADSORBING",
    BOUNCE_BACK: "BOUNCE_BACK"
    }

class BoundaryCondition(object):
    """
    Base class for boundary condition specifications.

    type_of_BC: specifies the boundary condition
    T: fixed wall temperature (in degrees K) that will be used if
        the boundary conditions needs such a value.
    P: used only for outflow conditions
    adsorb: (N pressures) x (N temperatures) x 3 numpy array for equilibrium isotherms
    beta_n: velocity aware adsorption probability, normal (0 = accept all, 1 = allow fast)
    beta_t: velocity aware adsorption probability, tangential (0 = accept all, 1 = allow slow)
    alpha_n: CercignaniLampis accommodation coefficient, normal (0 = specular, 1 = diffuse)
    alpha_t: CercignaniLampis accommodation coefficient, tangential (0 = specular, 1 = diffuse)
    k_f: adsorbing wall forward reaction rate, gets non-dimmed to gamma_f
    S_T: total adsorption sites on the wall, kg/m
    cover_initial: fraction of wall that is initially covered [0,1]
    alpha_p: constant for conversion to non-dimensional terms, see ugks_data
    gamma_f: see k_f
    reflect_type: select Cercignani-Lampis or specular reflection ['S','CL']
    """
    __slots__ = 'type_of_BC', 'D', 'T', "U", "V","P", 'other_block',\
            'other_face', 'orientation', 'label', 'UDF_U',\
            'UDF_V', 'UDF_T','UDF_D','adsorb', 'beta_n', 'beta_t', 'alpha_n', \
            'alpha_t', 'alpha_p', 'k_f', 'S_T', 'cover_initial','gamma_f',\
            'reflect_type','flip_distribution','transform','dynamic'
            
    def __init__(self,
                 type_of_BC = REFLECT,
                 D = 0.0,
                 T = 0.0,
                 U = 0.0,
                 V = 0.0,
                 P = 0.0,
                 UDF_D = None,
                 UDF_U = None,
                 UDF_V = None,
                 UDF_T = None,
                 adsorb = None,
                 beta_n=0,
                 beta_t=0,
                 alpha_n=1,
                 alpha_t=1,
                 k_f=0,
                 S_T=1.0,
                 cover_initial=0,
                 reflect_type='S',
                 dynamic=1,
                 other_block=-1,
                 other_face=-1,
                 orientation=NO_HOLD,
                 flip_distribution=NO_FLIP,
                 transform=TRANSFORM,
                 label=""):
                     
        self.type_of_BC = copy.copy(type_of_BC)
        self.D = copy.copy(D)
        self.T = copy.copy(T)
        self.U = copy.copy(U)
        self.V = copy.copy(V)
        self.P = copy.copy(P)
        self.UDF_D = copy.copy(UDF_D)
        self.UDF_U = copy.copy(UDF_U)
        self.UDF_V = copy.copy(UDF_V)
        self.UDF_T = copy.copy(UDF_T)
        if adsorb != None:
            self.adsorb = np.copy(adsorb)
        else:
            self.adsorb = None
        self.beta_n=copy.copy(beta_n)
        self.beta_t=copy.copy(beta_t)
        self.alpha_n=copy.copy(alpha_n)
        self.alpha_t=copy.copy(alpha_t)
        self.k_f=copy.copy(k_f)
        self.gamma_f=None
        self.S_T=copy.copy(S_T)
        self.cover_initial=copy.copy(cover_initial)
        self.reflect_type=copy.copy(reflect_type)
        self.dynamic=copy.copy(dynamic)
        self.alpha_p = 0
        self.other_block = copy.copy(other_block)
        self.other_face = copy.copy(other_face)
        self.orientation = copy.copy(orientation)
        self.flip_distribution = copy.copy(flip_distribution)
        self.transform = copy.copy(transform)
        self.label = copy.copy(label)
            
        return

    def __copy__(self):
        return BoundaryCondition(type_of_BC=self.type_of_BC,
                                 D=self.D,
                                 T=self.T,
                                 U=self.U,
                                 V=self.V,
                                 P=self.P,
                                 UDF_D = self.UDF_D,
                                 UDF_U = self.UDF_U,
                                 UDF_V = self.UDF_V,
                                 UDF_T = self.UDF_T,
                                 adsorb = self.adsorb,
                                 beta_n = self.beta_n,
                                 beta_t = self.beta_t,
                                 alpha_n = self.alpha_n,
                                 alpha_t = self.alpha_t,
                                 k_f = self.k_f,
                                 S_T = self.S_T,
                                 cover_initial = self.cover_initial,
                                 dynamic = self.dynamic,
                                 reflect_type=self.reflect_type,
                                 other_block=self.other_block,
                                 other_face=self.other_face,
                                 orientation=self.orientation,
                                 transform=self.transform,
                                 label=self.label)
    
class AdjacentBC(BoundaryCondition):
    """
    This boundary joins (i.e. is adjacent to) a boundary of another block.

    This condition is usually not set manually but is set as part of the
    connect_blocks() function.
    """
    def __init__(self, other_block=-1, other_face=-1, orientation=NO_HOLD, transform=TRANSFORM, label="ADJACENT"):
        BoundaryCondition.__init__(self, type_of_BC=ADJACENT, other_block=other_block,
                                   other_face=other_face, orientation=orientation,
                                   transform=transform, label=label)
        return
    def __str__(self):
        return "AdjacentBC(other_block=%d, other_face=%d, orientation=%d, label=\"%s\")" % \
            (self.other_block, self.other_face, self.orientation, self.label)
    def __copy__(self):
        return AdjacentBC(other_block=self.other_block,
                          other_face=self.other_face,
                          orientation=self.orientation,
                          transform=self.transform,
                          label=self.label)
class PeriodicBC(BoundaryCondition):
    """
    This boundary joins (i.e. is adjacent to) a boundary of another block.
    """
    def __init__(self, other_block=-1, other_face=-1,orientation=NO_HOLD, flip_distribution=NO_FLIP, transform=TRANSFORM, label="PERIODIC"):
        BoundaryCondition.__init__(self, type_of_BC=ADJACENT, other_block=other_block,
                                   other_face=other_face, orientation=orientation,
                                   flip_distribution=flip_distribution, transform=transform,
                                   label=label)
        return
    def __str__(self):
        return "PeriodicBC(other_block=%d, other_face=%d, label=\"%s\")" % \
            (self.other_block, self.other_face, self.label)
    def __copy__(self):
        return PeriodicBC(other_block=self.other_block,
                          other_face=self.other_face,
                          orientation=self.orientation,
                          flip_distribution=self.flip_distribution,
                          transform=self.transform,
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
                 UDF_U=None, UDF_V=None, UDF_T=None, flowCondition=None, label="CONSTANT"):
        if flowCondition != None:
            D = flowCondition.D
            T = flowCondition.T
            U = flowCondition.U
            V = flowCondition.V
            UDF_D = flowCondition.UDF_D
            UDF_U = flowCondition.UDF_U
            UDF_V = flowCondition.UDF_V
            UDF_T = flowCondition.UDF_T
        BoundaryCondition.__init__(self,D=D, U=U, V=V, T=T, 
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
        
class BounceBackBC(BoundaryCondition):
    """
    fill the ghost cells with data that is a double reflection of the data in the flow 
    domain about the plane of the boundary
    """
    def __init__(self, label="BOUNCE_BACK"):
        BoundaryCondition.__init__(self, type_of_BC=BOUNCE_BACK, label=label)
        return
        
    def __str__(self):
        return "BounceBackBC(label=\"%s\")" % self.label
    def __copy__(self):
        return BounceBackBC(label=self.label)

class DiffuseBC(BoundaryCondition):
    """
    define the values used to define the equilibrium distribution for the wall
    U = velocity of wall, parallel to wall, positive in direction of cell wall tangent vector
    """
    def __init__(self, D=1, T=0, U=0, V=0, UDF_D=None, 
                 UDF_U=None, UDF_V=None, UDF_T=None, flowCondition=None, 
                 label="DIFFUSE"):
                     
        if flowCondition != None:
            D = flowCondition.D
            T = flowCondition.T
            U = flowCondition.U
            V = flowCondition.V
            UDF_D = flowCondition.UDF_D
            UDF_U = flowCondition.UDF_U
            UDF_V = flowCondition.UDF_V
            UDF_T = flowCondition.UDF_T
            
        BoundaryCondition.__init__(self, D=D, U=U, 
                                   V=V, T=T, UDF_D=UDF_D, 
                                   UDF_U=UDF_U, UDF_V=UDF_V, UDF_T=UDF_T,
                                   type_of_BC=DIFFUSE, label=label)
        
        return
    def __str__(self):
        return "DiffuseBC(D=%g, T=%g, U=%g, V=%g, UDF_D=%s, UDF_U=%s, UDF_V=%s, UDF_T=%s, label=\"%s\")" % \
            (self.D, self.T, self.U, self.V, self.UDF_D, self.UDF_U, self.UDF_V, self.UDF_T, self.label)
    def __copy__(self):
        return DiffuseBC(D=self.D,
                               T = self.T,
                               U = self.U, 
                               V = self.V,
                               UDF_D = self.UDF_D,
                               UDF_U = self.UDF_U, 
                               UDF_V = self.UDF_V,
                               UDF_T = self.UDF_T,
                               label=self.label)
                               
class AdsorbingBC(BoundaryCondition):
    """
    define the values used to define the equilibrium distribution for the wall
    U = velocity of wall, parallel to wall, positive in direction of cell wall tangent vector
    """
    def __init__(self, D=1, T=0, U=0, V=0, UDF_D=None, 
                 UDF_U=None, UDF_V=None, UDF_T=None, 
                 flowCondition=None, 
                 adsorb=None,
                 beta_n=0,
                 beta_t=0,
                 alpha_n=1,
                 alpha_t=1,
                 k_f=0,
                 S_T=1.0,
                 cover_initial=0,
                 reflect_type='S',
                 dynamic=1,
                 label="ADSORBING"):
                     
        if flowCondition != None:
            D = flowCondition.D
            T = flowCondition.T
            U = flowCondition.U
            V = flowCondition.V
            UDF_D = flowCondition.UDF_D
            UDF_U = flowCondition.UDF_U
            UDF_V = flowCondition.UDF_V
            UDF_T = flowCondition.UDF_T
            
        BoundaryCondition.__init__(self, D=D, U=U, 
                                   V=V, T=T, UDF_D=UDF_D, 
                                   UDF_U=UDF_U, UDF_V=UDF_V, UDF_T=UDF_T,
                                   adsorb=adsorb, beta_n=beta_n, beta_t=beta_t,
                                   alpha_n=alpha_n, alpha_t=alpha_t,
                                   k_f=k_f, S_T=S_T, cover_initial=cover_initial,
                                   reflect_type=reflect_type,dynamic=dynamic,
                                   type_of_BC=ADSORBING, label=label)
        
        return
    def __copy__(self):
        return AdsorbingBC(D=self.D,
                            T = self.T,
                            U = self.U, 
                            V = self.V,
                            UDF_D = self.UDF_D,
                            UDF_U = self.UDF_U, 
                            UDF_V = self.UDF_V,
                            UDF_T = self.UDF_T,
                            adsorb = self.adsorb,
                            beta_n = self.beta_n,
                            beta_t = self.beta_t,
                            alpha_n = self.alpha_n,
                            alpha_t = self.alpha_t,
                            k_f = self.k_f,
                            S_T = self.S_T,
                            cover_initial = self.cover_initial,
                            reflect_type=self.reflect_type,
                            dynamic=self.dynamic,
                            label=self.label)
                               
class InflowBC(BoundaryCondition):
    """
    populate the ghost cell with an equilibrium distribution with defined 
    density and temperature, but with mean velocity normal to the wall
    defined according to the adjacent cell
    """
    def __init__(self, D=1, T=1, label="INFLOW"):
        BoundaryCondition.__init__(self, D=D, T=T, type_of_BC=INFLOW,
                                   label=label)
        return
    def __str__(self):
        return "InflowBC(D=%f, T=%f, label=\"%s\")" %(self.D, self.T, self.label)
    def __copy__(self):
        return InflowBC(D=self.D, T=self.T, label=self.label)

class OutflowBC(BoundaryCondition):
    """
    populate the ghost cell with an equilibrium distribution with defined 
    pressure. This means we calculate temperature and mean velocity normal to the wall
    according to the adjacent cell and adjust the density accordingly
    """
    def __init__(self, P=1, label="OUTFLOW"):
        BoundaryCondition.__init__(self, P=P, type_of_BC=OUTFLOW,
                                   label=label)
        return
    def __str__(self):
        return "OutflowBC(P=%f, label=\"%s\")" %(self.P, self.label)
    def __copy__(self):
        return OutflowBC(P=self.P, label=self.label)