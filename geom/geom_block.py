"""@package e3_block
\brief Classes for defining 2D and 3D blocks.

It is expected that this module be imported by the application program e3prep.py.

\ingroup eilmer3
\author PJ
\version 16-Mar-2008: extracted from e3prep.py (formerly mbcns_prep.py)
\version 29-Nov-2010: moved face and vertex definitions to separate file.
"""

from pygeom.pygeom import *
from geom_grid import *
import copy

# Dictionaries to look up face index values from name or number.
from geom_defs import *
from geom_bc_defs import *

#----------------------------------------------------------------------

def make_patch(north, east, south, west, grid_type="TFI"):
    """
    Defines a 2D patch (or region) by its boundary paths (in order NESW).

    A patch is defined by its four bounding L{Path} objects
    with assumed positive directions as shown::

        .               NORTH
        . 1       +------->-------+
        . |       |               |
        . s  WEST |               | EAST
        . |       |               |
        . 0       +------->-------+
        .               SOUTH
        .
        .         0-------r------>1

    NORTH and SOUTH boundaries progress WEST to EAST while
    EAST and WEST boundaries progress SOUTH to NORTH.
    To reuse a L{Path} object when building multiple blocks,
    you will need to pay attention to the orientation of the blocks
    and the defined positive direction of the L{Path} object.

    north: bounding path on the NORTH side
    east: bounding path on the EAST side
    south: bounding path on the SOUTH side
    west: bounding path on the WEST side
    grid_type: indicates the type of interpolation within the patch
        TFI, COONS: transfinite interpolation or Coons patch
        AO: interpolation via an area-orthogonality grid, used as background
    """
    if not isinstance(north, Path):
        raise TypeError, ("north should be a Path but it is: %s" % type(north))
    if not isinstance(east, Path):
        raise TypeError, ("east should be a Path but it is: %s" % type(east))
    if not isinstance(south, Path):
        raise TypeError, ("south should be a Path but it is: %s" % type(south))
    if not isinstance(west, Path):
        raise TypeError, ("west should be a Path but it is: %s" % type(west))
    grid_type.upper()
    if grid_type == "AO":
        return AOPatch(south, north, west, east)
    else:
        return CoonsPatch(south, north, west, east)

#----------------------------------------------------------------------------

def close_enough(vA, vB, tolerance=1.0e-4):
    """
    Decide if two Vector quantities are close enough to being equal.

    This will be used to test that the block corners coincide.
    """
    return (vabs(vA - vB)/(vabs(vA + vB)+1.0)) <= tolerance

#----------------------------------------------------------------------------

class Block(object):
    """
    Python base class to organise the setting of block parameters.

    We will create instances of its subclasses: Block2D or Block3D.
    """
    # We will accumulate references to defined objects.
    blockList = []
    connectionList = []
    # Minimum number of cells in any direction
    # The boundary conditions affect two cells into the mesh.
    nmin = 2

    def set_BC(self, face_name, type_of_BC, D = 0.0, U = 0.0, 
               V = 0.0, T=0.0,UDF_D=None, UDF_U=None, UDF_V=None, 
               UDF_T=None, label="", flowCondition = None, other_block = None,
               other_face = None):
        """
        Sets a boundary condition on a particular face of the block.

        Sometimes it is good to be able to adjust properties after
        block creation; this function provides that capability.

        face_name: int identifier to select the appropriate boundary within the block.
        type_of_BC: Name or index value of the requested boundary
            condition:
                0 or ADJACENT: block interface
                1 or OUTFLOW: zeroth order extrapolation
                2 or CONST: defined conditions that do not change for the duration
                3 or REFLECT: inviscid reflecting wall
                4 or ACCOMMODATING: accommodating wall
        D: density of wall
        U: velocity of wall, (x direction if for CONSTANT, else parallel to boundary)
        V: velocity of wall in y
        T: If appropriate, specify the boundary-wall temperature in degrees Kelvin.
        """
        iface = faceDict[face_name]
        type_of_BC = bcIndexFromName[str(type_of_BC).upper()]
        print "Set block:", self.blkId, "face:", faceName[iface], \
              "BC:", bcName[type_of_BC]
        
        D = float(D)
        U = float(U)
        V = float(V)
        T = float(T)
        
            
        # Now, create a new boundary condition object.
        if type_of_BC == ADJACENT:
            print "Error in setting BC:"
            print "    Should not be setting an ADJACENT BC on a single block."
            return
        if type_of_BC == EXTRAPOLATE_OUT:
            newbc = ExtrapolateOutBC(label=label)
        if type_of_BC == CONSTANT:
            if flowCondition:
                newbc = ConstantBC(D=flowCondition.D,
                                  T = flowCondition.T,
                                  U = flowCondition.U,
                                  V = flowCondition.V,
                                  UDF_D = flowCondition.UDF_D,
                                  UDF_U = flowCondition.UDF_U,
                                  UDF_V = flowCondition.UDF_V,
                                  UDF_T = flowCondition.UDF_T,
                                   label = flowCondition.label)
            else:
                newbc = ConstantBC(D=D, T=T, U=U, V=V, UDF_D=UDF_D, 
                                   UDF_U=UDF_U, UDF_V=UDF_V, UDF_T=UDF_T, 
                                   label=label)
        if type_of_BC == INFLOW:
            if flowCondition:
                newbc = InflowBC(D=flowCondition.D,
                                  T = flowCondition.T,
                                   label = flowCondition.label)
            else:
                newbc = ConstantBC(D=D, T=T, label=label)
                
        if type_of_BC == OUTFLOW:
            if flowCondition:
                newbc = OutflowBC(P=flowCondition.P, label = flowCondition.label)
            else:
                newbc = ConstantBC(P=P, label=label)
                
        if type_of_BC == REFLECT:
            newbc = ReflectBC(label=label)
        if type_of_BC == ACCOMMODATING:
            if flowCondition:
                newbc = AccommodatingBC(D=flowCondition.D,
                                  T = flowCondition.T,
                                  U = flowCondition.U,
                                  V = flowCondition.V,
                                  UDF_D = flowCondition.UDF_D,
                                  UDF_U = flowCondition.UDF_U,
                                  UDF_V = flowCondition.UDF_V,
                                  UDF_T = flowCondition.UDF_T,
                                  label = flowCondition.label)
            else:
                newbc = AccommodatingBC(D=D, T=T, U=U, 
                                        V=V, UDF_D=UDF_D, UDF_U=UDF_U, 
                                        UDF_V=UDF_V, UDF_T=UDF_T, label=label)
        if type_of_BC == PERIODIC:
            newbc = PeriodicBC(other_block.blkId, other_face, label = label)
            other_block.bc_list[other_face] = PeriodicBC(self.blkId, iface, label = label)
        try:
            self.bc_list[iface] = newbc
        except:
            print "Boundary condition not set correctly, type_of_BC=", type_of_BC
            sys.exit()
        return
#----------------------------------------------------------------------------

class Block2D(Block):
    """
    Python class to organise the setting of block parameters for 2D flow.
    """

    __slots__ = 'blkId', 'label', 'psurf', 'nni', 'nnj', 'nnk', 'grid', \
                'fill_condition', 'active', \
                'cf_list', 'bc_list', 'vtx', \
                
    def __init__(self,
                 psurf=None,
                 grid=None,
                 import_grid_file_name=None,
                 nni=2,
                 nnj=2,
                 cf_list=[None,]*4,
                 bc_list=[ReflectBC(),]*4,
                 fill_condition=None,
                 label="",
                 active=1
                 ):
        """
        Create a block from a parametric-surface object.

        You should specify on of the following three:
        psurf: The ParametricSurface object defining the block in space.
            Typically, this will be a CoonsPatch or an AOPatch.
        grid: A StructuredGrid object may be supplied directly.
        import_grid_file_name: name of a VTK file containing the grid

        nni: number of cells in the i-direction (west to east)
        nnj: number of cells in the j-direction (south to north)
        cf_list: List of the cluster_function objects, one for each boundary.
            The order within the list is NORTH, EAST, SOUTH and WEST.
        bc_list: List of BoundaryCondition objects, one for each face.
        fill_condition: Either a single FlowCondition or user-defined function.

        Note: The blocks are given their identity (counting from zero)
            according to the order in which they are created by the user's script.
            This identity is stored as blkId and is used internally
            in the preprocessing, simulation and postprocessing stages.
        """
        if not isinstance(nni, int):
            raise TypeError, ("nni should be an int but it is: %s" % type(nni))
        if not isinstance(nnj, int):
            raise TypeError, ("nnj should be an int but it is: %s" % type(nnj))
        #
        self.blkId = len(Block.blockList)    # next available block index
        if len(label) == 0:
            label = "blk-" + str(self.blkId)
        self.label = label
        self.active = active
        #
        # The grid may come from one of several sources, in order of priority:
        # 1. discretization of a parametric surface
        # 2. supplied directly as a StructuredGrid object
        # 3. read from a VTK file.
        print "Block2D.__init__: blkId=%d, label=%s" % (self.blkId, self.label)
        if psurf != None:
            print "Generate a grid from a user-specified parametric surface."
            self.psurf = psurf.clone() # needed for later rendering
            self.nni = nni
            self.nnj = nnj
            assert len(cf_list) == 4
            self.grid = StructuredGrid((self.nni+1, self.nnj+1))
            self.grid.make_grid_from_surface(self.psurf, cf_list)
        elif grid != None:
            print "Accept grid as given."
            # Should assert grid is StructuredGrid or e3_grid.StructuredGrid
            self.psurf = None # will be a problem for later rendering
            self.grid = grid
            self.nni = self.grid.ni - 1
            self.nnj = self.grid.nj - 1
        elif import_grid_file_name != None:
            print "Import a grid from a VTK data file:", import_grid_file_name
            self.grid = StructuredGrid()
            fin = open(import_grid_file_name, "r")
            self.grid.read_block_in_VTK_format(fin)
            fin.close()
            self.nni = self.grid.ni - 1
            self.nnj = self.grid.nj - 1
        else:
            raise ValueError("Block2D constructor was expecting one of psurf or grid.")
        self.vtx = [self.grid.get_vertex_coords(ivtx) for ivtx in range(4)]
        #
        assert self.nni >= Block.nmin
        assert self.nnj >= Block.nmin
        self.nnk = 1
        assert len(bc_list) == 4
        
        # Make copies of supplied lists in case we are given the same
        # (default) empty list for each block
        self.bc_list = copy.copy(bc_list)
        self.cf_list = copy.copy(cf_list)
        self.fill_condition = fill_condition
        #
        Block.blockList.append(self)
        return

    def cell_centre_location(self, i, j, k, gdata):
        """
        Return the cell geometry.

        Geometry of cell
        ^ j
        |
        |
        NW-----NE
        |       |
        |       |
        SW-----SE  --> i
        """
        # Should match the code in C++ function calc_volumes_2D() in block.cxx.
        k = 0
        xSE = self.grid.x[i+1,j,k];   ySE = self.grid.y[i+1,j,k]
        xNE = self.grid.x[i+1,j+1,k]; yNE = self.grid.y[i+1,j+1,k]
        xNW = self.grid.x[i,j+1,k];   yNW = self.grid.y[i,j+1,k]
        xSW = self.grid.x[i,j,k];     ySW = self.grid.y[i,j,k]
        # Cell area in the (x,y)-plane.
        xyarea = 0.5 * ((xNE + xSE) * (yNE - ySE) + (xNW + xNE) * (yNW - yNE) +
                        (xSW + xNW) * (ySW - yNW) + (xSE + xSW) * (ySE - ySW))
        # Cell Centroid.
        centre_x = 1.0 / (xyarea * 6.0) * \
            ((yNE - ySE) * (xSE * xSE + xSE * xNE + xNE * xNE) + 
             (yNW - yNE) * (xNE * xNE + xNE * xNW + xNW * xNW) +
             (ySW - yNW) * (xNW * xNW + xNW * xSW + xSW * xSW) + 
             (ySE - ySW) * (xSW * xSW + xSW * xSE + xSE * xSE))
        centre_y = -1.0 / (xyarea * 6.0) * \
            ((xNE - xSE) * (ySE * ySE + ySE * yNE + yNE * yNE) + 
             (xNW - xNE) * (yNE * yNE + yNE * yNW + yNW * yNW) +
             (xSW - xNW) * (yNW * yNW + yNW * ySW + ySW * ySW) + 
             (xSE - xSW) * (ySW * ySW + ySW * ySE + ySE * ySE))
        #
        if gdata.axisymmetric_flag == 1:
            # volume per radian
            vol = xyarea * centre_y
        else:
            # volume per unit depth in z
            vol = xyarea
        #
        return (centre_x, centre_y, 0.0, vol)
    
def connect_blocks_2D(A, faceA, B, faceB, with_udf=0, 
                      filename=None, is_wall=0, use_udf_flux=0):
    """
    Make the face-to-face connection between neighbouring blocks.

    A: first Block2D object
    faceA: indicates which face of block A is to be connected.
        The constants NORTH, EAST, SOUTH, and WEST may be convenient to use.
    B: second Block2D object
    faceB: indicates which face of block B is to be connected.
        The constants NORTH, EAST, SOUTH, and WEST may be convenient to use.
    """
    assert isinstance(A, Block2D)
    assert isinstance(B, Block2D)
    assert faceA in faceList2D
    assert faceB in faceList2D
    print "connect block", A.blkId, "face", faceName[faceA], \
          "to block", B.blkId, "face", faceName[faceB]
          
    # save connection to list
    Block.connectionList.append([A.blkId, faceA, B.blkId, faceB])
      
    if with_udf:
        # Exchange connection with user-defined function.
        A.bc_list[faceA] = AdjacentPlusUDFBC(B.blkId, faceB, filename=filename, 
                                             is_wall=is_wall, use_udf_flux=use_udf_flux)
        B.bc_list[faceB] = AdjacentPlusUDFBC(A.blkId, faceA, filename=filename, 
                                             is_wall=is_wall, use_udf_flux=use_udf_flux)
    else:
        # Classic exchange connection.
        A.bc_list[faceA] = AdjacentBC(B.blkId, faceB)
        B.bc_list[faceB] = AdjacentBC(A.blkId, faceA)
    return


def identify_block_connections_2D(block_list=None, exclude_list=[], tolerance=1.0e-6):
    """
    Identifies and makes block connections based on block-vertex locations.

    block_list: list of Block2D objects that are to be included in the search.
        If none is supplied, the whole collection is searched.
        This allows one to specify a limited selection of blocks
        to be connected.
    exclude_list: list of pairs of Block2D objects that should not be connected
    tolerance: spatial tolerance for colocation of vertices
    """
    if block_list == None:
        # Default to searching all defined blocks.
        block_list = Block.blockList
    print "Begin searching for block connections..."
    for thisBlock in block_list:
        for otherBlock in block_list:
            if thisBlock == otherBlock: continue
            inExcludeList = (exclude_list.count((thisBlock,otherBlock)) + \
                             exclude_list.count((otherBlock,thisBlock))) > 0
            if not inExcludeList:
                connections = 0
                #
                if (vabs(thisBlock.vtx[NE] - otherBlock.vtx[SW]) < tolerance) and \
                   (vabs(thisBlock.vtx[NW] - otherBlock.vtx[NW]) < tolerance) :
                    connect_blocks_2D(thisBlock, NORTH, otherBlock, WEST)
                    connections += 1
                if (vabs(thisBlock.vtx[NE] - otherBlock.vtx[NW]) < tolerance) and \
                   (vabs(thisBlock.vtx[NW] - otherBlock.vtx[NE]) < tolerance) :
                    connect_blocks_2D(thisBlock, NORTH, otherBlock, NORTH)
                    connections += 1
                if (vabs(thisBlock.vtx[NE] - otherBlock.vtx[NE]) < tolerance) and \
                   (vabs(thisBlock.vtx[NW] - otherBlock.vtx[SE]) < tolerance) :
                    connect_blocks_2D(thisBlock, NORTH, otherBlock, EAST)
                    connections += 1
                if (vabs(thisBlock.vtx[NE] - otherBlock.vtx[SE]) < tolerance) and \
                   (vabs(thisBlock.vtx[NW] - otherBlock.vtx[SW]) < tolerance) :
                    connect_blocks_2D(thisBlock, NORTH, otherBlock, SOUTH)
                    connections += 1
                #
                if (vabs(thisBlock.vtx[SE] - otherBlock.vtx[SW]) < tolerance) and \
                   (vabs(thisBlock.vtx[NE] - otherBlock.vtx[NW]) < tolerance) :
                    connect_blocks_2D(thisBlock, EAST, otherBlock, WEST)
                    connections += 1
                if (vabs(thisBlock.vtx[SE] - otherBlock.vtx[NW]) < tolerance) and \
                   (vabs(thisBlock.vtx[NE] - otherBlock.vtx[NE]) < tolerance) :
                    connect_blocks_2D(thisBlock, EAST, otherBlock, NORTH)
                    connections += 1
                if (vabs(thisBlock.vtx[SE] - otherBlock.vtx[NE]) < tolerance) and \
                   (vabs(thisBlock.vtx[NE] - otherBlock.vtx[SE]) < tolerance) :
                    connect_blocks_2D(thisBlock, EAST, otherBlock, EAST)
                    connections += 1
                if (vabs(thisBlock.vtx[SE] - otherBlock.vtx[SE]) < tolerance) and \
                   (vabs(thisBlock.vtx[NE] - otherBlock.vtx[SW]) < tolerance) :
                    connect_blocks_2D(thisBlock, EAST, otherBlock, SOUTH)
                    connections += 1
                #
                if (vabs(thisBlock.vtx[SW] - otherBlock.vtx[SW]) < tolerance) and \
                   (vabs(thisBlock.vtx[SE] - otherBlock.vtx[NW]) < tolerance) :
                    connect_blocks_2D(thisBlock, SOUTH, otherBlock, WEST)
                    connections += 1
                if (vabs(thisBlock.vtx[SW] - otherBlock.vtx[NW]) < tolerance) and \
                   (vabs(thisBlock.vtx[SE] - otherBlock.vtx[NE]) < tolerance) :
                    connect_blocks_2D(thisBlock, SOUTH, otherBlock, NORTH)
                    connections += 1
                if (vabs(thisBlock.vtx[SW] - otherBlock.vtx[NE]) < tolerance) and \
                   (vabs(thisBlock.vtx[SE] - otherBlock.vtx[SE]) < tolerance) :
                    connect_blocks_2D(thisBlock, SOUTH, otherBlock, EAST)
                    connections += 1
                if (vabs(thisBlock.vtx[SW] - otherBlock.vtx[SE]) < tolerance) and \
                   (vabs(thisBlock.vtx[SE] - otherBlock.vtx[SW]) < tolerance) :
                    connect_blocks_2D(thisBlock, SOUTH, otherBlock, SOUTH)
                    connections += 1
                #
                if (vabs(thisBlock.vtx[NW] - otherBlock.vtx[SW]) < tolerance) and \
                   (vabs(thisBlock.vtx[SW] - otherBlock.vtx[NW]) < tolerance) :
                    connect_blocks_2D(thisBlock, WEST, otherBlock, WEST)
                    connections += 1
                if (vabs(thisBlock.vtx[NW] - otherBlock.vtx[NW]) < tolerance) and \
                   (vabs(thisBlock.vtx[SW] - otherBlock.vtx[NE]) < tolerance) :
                    connect_blocks_2D(thisBlock, WEST, otherBlock, NORTH)
                    connections += 1
                if (vabs(thisBlock.vtx[NW] - otherBlock.vtx[NE]) < tolerance) and \
                   (vabs(thisBlock.vtx[SW] - otherBlock.vtx[SE]) < tolerance) :
                    connect_blocks_2D(thisBlock, WEST, otherBlock, EAST)
                    connections += 1
                if (vabs(thisBlock.vtx[NW] - otherBlock.vtx[SE]) < tolerance) and \
                   (vabs(thisBlock.vtx[SW] - otherBlock.vtx[SW]) < tolerance) :
                    connect_blocks_2D(thisBlock, WEST, otherBlock, SOUTH)
                    connections += 1
                #
                if connections > 0:
                    # Avoid doubling-up with the reverse connections.
                    exclude_list.append((thisBlock,otherBlock))
    print "Finished searching for block connections."
    return

# --------------------------------------------------------------------

class MultiBlock2D(object):
    """
    Allows us to specify a block of sub-blocks.

    A number of internally-connected Block2D objects will be created when
    one MultiBlock2D object is created.
    Individual blocks occupy subsections of the original parametric surface.
   
    Note that the collection of Block2D objects will be stored in
    a list of lists with each inner-list storing a j-column of blocks::
    
        .                       North
        .   1       +-------------+-------------+
        .   |       |   [0][1]    |   [1][1]    |
        .   s  West +-------------+-------------+ East
        .   |       |   [0][0]    |   [1][0]    |
        .   0       +-------------+-------------+
        .                       South
        .           0           --r-->          1

    The user script may access an individual block within the MultiBlock2D
    object as object.blks[i][j].
    This will be useful for connecting blocks within the MultiBlock cluster
    to other blocks as defined in the user's script.

    Some properties, such as fill_conditions and grid_type, will be propagated
    to all sub-blocks.  Individual sub-blocks can be later customised.
    """

    __slots__ = 'psurf', 'bc_list', 'nb_w2e', 'nb_s2n', 'nn_w2e',\
                'nn_s2n', 'cluster_w2e', 'cluster_s2n', 'fill_condition', \
                'grid_type', 'split_single_grid', 'label', 'blks'
    
    def __init__(self,
                 psurf=None,
                 nni=None,
                 nnj=None,
                 bc_list=[ReflectBC(), ReflectBC(), ReflectBC(), ReflectBC()],
                 nb_w2e=1,
                 nb_s2n=1,
                 nn_w2e=None,
                 nn_s2n=None,
                 cluster_w2e=None,
                 cluster_s2n=None,
                 fill_condition=None,
                 label="blk",
                 active=1):
        """
        Create a cluster of blocks within an original parametric surface.

        psurf: ParametricSurface which defines the block.
        bc_list: List of boundary condition objects
            The order within the list is NORTH, EAST, SOUTH and WEST.
        nb_w2e: Number of sub-blocks from west to east.
        nb_s2n: Number of sub-blocks from south to north.
        nn_w2e: List of discretisation values for north and south
            boundaries of the sub-blocks.
            If a list is not supplied, the original number of cells for the
            outer boundary is divided over the individual sub-block boundaries.
        nn_s2n: List of discretisation values for west and east
            boundaries of the sub-blocks.
            If a list is not supplied, the original number of cells for the
            outer boundary is divided over the individual sub-block boundaries.
        cluster_w2e: If a list of cluster functions is supplied,
            individual clustering will be applied to the corresponding
            south and north boundaries of each sub-block.
            If not supplied, a default of no clustering will be applied.
        cluster_s2n: If a list of cluster functions is supplied,
            individual clustering will be applied to the corresponding
            west and east boundaries of each sub-block.
            If not supplied, a default of no clustering will be applied.
        fill_condition: A single FlowCondition object that is to be
            used for all sub-blocks
        grid_type: Select the type of grid generator from TFI or AO.
        split_single_grid : If this boolean flag is true, a single grid is
            generated which is then subdivided into the required blocks.
        label: A label that will be augmented with the sub-block index
            and then used to label the individual Block2D objects.
        active: =1 (default) the time integration operates for this block
                =0 time integration for this block is suppressed
        """
        default_cluster_function = LinearFunction()
        self.blks = []
        dt_s2n = 1.0 / nb_s2n
        dt_w2e = 1.0 / nb_w2e
        for i in range(nb_w2e):
            self.blks.append([])  # for a new j-column of blocks
            for j in range(nb_s2n):
                sublabel = label + "-" + str(i) + "-" + str(j)
                new_psurf = psurf.clone()
                #
                new_psurf.s0 = j * dt_s2n
                new_psurf.s1 = (j+1) * dt_s2n
                new_psurf.r0 = i * dt_w2e
                new_psurf.r1 = (i+1) * dt_w2e
                #
                # Transfer the boundary conditions.
                bc_newlist=[ReflectBC(), ReflectBC(), ReflectBC(), ReflectBC()]
                if i == 0:
                    bc_newlist[WEST] = copy.copy(bc_list[WEST])
                if i == nb_w2e - 1:
                    bc_newlist[EAST] = copy.copy(bc_list[EAST])
                if j == 0:
                    bc_newlist[SOUTH] = copy.copy(bc_list[SOUTH])
                if j == nb_s2n - 1:
                    bc_newlist[NORTH] = copy.copy(bc_list[NORTH])
                #
                # For discretization, take the list of cell numbers
                # if it available, else divide overall number of nodes.
                if nn_s2n:
                    new_nnj = nn_s2n[j]
                else:
                    new_nnj = nnj / nb_s2n
                if nn_w2e:
                    new_nni = nn_w2e[i]
                else:
                    new_nni = nni / nb_w2e
                #
                cf_newlist = []
                try:
                    cf_newlist.append(cluster_w2e[i])
                except:
                    cf_newlist.append(default_cluster_function)
                try:
                    cf_newlist.append(cluster_s2n[j])
                except:
                    cf_newlist.append(default_cluster_function)
                try:
                    cf_newlist.append(cluster_w2e[i])
                except:
                    cf_newlist.append(default_cluster_function)
                try:
                    cf_newlist.append(cluster_s2n[j])
                except:
                    cf_newlist.append(default_cluster_function)
                #
                new_blk = Block2D(psurf=new_psurf, nni=new_nni, nnj=new_nnj,
                                  cf_list=cf_newlist, bc_list=bc_newlist,
                                  fill_condition=fill_condition,
                                  label=sublabel, active=active)
                self.blks[i].append(new_blk)
        #
        # print "blocks:", self.blks
        # Make the internal, sub-block to sub-block connections.
        if nb_w2e > 1:
            for i in range(1,nb_w2e):
                for j in range(nb_s2n):
                    connect_blocks_2D(self.blks[i-1][j], EAST, self.blks[i][j], WEST)
        if nb_s2n > 1:
            for i in range(nb_w2e):
                for j in range(1,nb_s2n):
                    connect_blocks_2D(self.blks[i][j-1], NORTH, self.blks[i][j], SOUTH)
        #
        return
    
# --------------------------------------------------------------------

def subdivide_vertex_range(ncells, nblocks):
    """
    Subdivide one index-direction into a number of sub-blocks within a SuperBlock.

    Input:
    ncells  : number of cells in one index-direction
    nblocks : number of sub-blocks in the same direction
    
    Returns a list of tuples specifying the subranges of the vertices that form the sub-blocks.
    """
    vtx_subranges = []
    vtx_min = 0
    for iblk in range(nblocks):
        n_subblock = int(ncells / (nblocks - iblk))  # try to divide remaining cells uniformly
        vtx_max = vtx_min + n_subblock
        vtx_subranges.append((vtx_min, vtx_max))
        vtx_min = vtx_max
        ncells -= n_subblock # remaining cells to be sub-divided
    return vtx_subranges

class SuperBlock2D(object):
    """
    Creates a single grid over the region and then subdivides that grid.

    Original implementation by Rowan; refactored (lightly) by PJ Nov-2010.
    """

    __slots__ = 'psurf', 'bc_list', 'nni', 'nnj', 'nbi', 'nbj', 'cf_list',\
                'fill_condition', 'label', 'blks'
    
    def __init__(self,
                 psurf=None,
                 nni=2,
                 nnj=2,
                 nbi=1,
                 nbj=1,
                 cf_list=[None, None, None, None],
                 bc_list=[ReflectBC(), ReflectBC(), ReflectBC(), ReflectBC()],
                 fill_condition=None,
                 label="sblk",
                 active=1
                 ):
        """
        Creates a single grid over the region and then subdivides that grid.

        On return, self.blks is a (nested) list-of-lists of subblock references.
        """
        self.blks = []
        # 1. Create the large grid for the super-block
        grid = StructuredGrid((nni+1, nnj+1))
        grid.make_grid_from_surface(psurf, cf_list)
        # 2. Create lists of the subgrid indices
        si_list = subdivide_vertex_range(nni, nbi)
        sj_list = subdivide_vertex_range(nnj, nbj)
        # 3. Do the actual subdivision.
        for i in range(nbi):
            self.blks.append([])  # for a new j-column of blocks
            for j in range(nbj):
                sublabel = label + "-" + str(i) + "-" + str(j)
                imin = si_list[i][0]; imax = si_list[i][1]
                jmin = sj_list[j][0]; jmax = sj_list[j][1]
                subgrid = grid.create_subgrid(imin, imax, jmin, jmax)
                #
                # Transfer the boundary conditions.
                bc_newlist=[ReflectBC(), ReflectBC(), ReflectBC(), ReflectBC()]
                if i == 0:
                    bc_newlist[WEST] = copy.copy(bc_list[WEST])
                if i == nbi - 1:
                    bc_newlist[EAST] = copy.copy(bc_list[EAST])
                if j == 0:
                    bc_newlist[SOUTH] = copy.copy(bc_list[SOUTH])
                if j == nbj - 1:
                    bc_newlist[NORTH] = copy.copy(bc_list[NORTH])
                #
                new_blk = Block2D(grid=subgrid, bc_list=bc_newlist,
                                  nni=imax-imin, nnj=jmax-jmin, 
                                  fill_condition=fill_condition,
                                  label=sublabel, active=active)
                self.blks[i].append(new_blk)
        # print "blocks:", self.blks
        #
        # 4. Make the internal, sub-block to sub-block connections.
        if nbi > 1:
            for i in range(1,nbi):
                for j in range(nbj):
                    connect_blocks_2D(self.blks[i-1][j], EAST, self.blks[i][j], WEST)
        if nbj > 1:
            for i in range(nbi):
                for j in range(1,nbj):
                    connect_blocks_2D(self.blks[i][j-1], NORTH, self.blks[i][j], SOUTH)
        #
        return
    
# --------------------------------------------------------------------

def identify_colocated_vertices(A, B, tolerance):
    """
    Identify colocated vertices by looking at their position is 3D space.

    A: Block3D object
    B: Block3D object
    tolerance : Vertices are considered to be colocated if their Euclidian distance
        is less than tolerance.
    """
    #from math import sqrt
    vtxPairList = []
    for iA in range(8):
        for iB in range(8):
            if vtxPairList.count((iA,iB)) > 0: continue
            if vabs(A.vtx[iA] - B.vtx[iB]) <= tolerance: vtxPairList.append((iA,iB))
    return vtxPairList


def identify_block_connections(block_list=None, exclude_list=[], tolerance=1.0e-6):
    """
    Identifies and makes block connections based on vertex locations.

    block_list   : list of Block2D objects that are to be included in the search.
       If none is supplied, the whole collection is searched.
       This allows one to specify a limited selection of blocks to be connected.
    exclude_list : list of pairs of Block3D objects that should not be connected
    tolerance    : spatial tolerance for colocation of vertices

    Note that this function is just a proxy for the specialized 2D and 3D functions.
    """
    identify_block_connections_2D(block_list, exclude_list, tolerance)
    return
