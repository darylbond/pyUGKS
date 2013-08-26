#ugks_block.py

"""
This file contains the definition of an object that holds all the data required
 for the simulation of one block.
 NOTE: All values used in the core calculations are in NON-DIMENSIONAL form.
     The reference values used can be found in the gdata object in global_data
"""

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import time
import sys
import os

from ugks_data import Block, gdata
from ugks_CL import genOpenCL, update_source
from geom.geom_bc_defs import *
from geom.geom_defs import *

def m_tuple(x, y):
    """
    multiply two tuples or lists, return a tuple
    """
    return tuple(a*b for a, b in zip(x, y)), y
    
def size_cl(global_size, work_size):
    """
    generate an appropriate global size
    """

    gsize = []
    for i, s in enumerate(global_size):
        w = work_size[i]
        gsize.append(int(int(s/w)*w + np.sign(s%w)*w))
    
    return tuple(gsize), tuple(work_size)

class UGKSBlock(object):
    """
    A block that contains all required information for the simulation of 
    a single block.
    """
    
    blockList = []
    
    host_update = 0 # flag to indicate if the host is up to date
    macro_update = 0
    has_accommodating = False  #flag to indicate if we have any accommodating walls
    grab_timer = 0
    
    def __init__(self, block_id, cl_ctx, cl_queue):
        """
        generate instance of the simBlock object from global data and 
        a block number. The global data is imported as a module and is 
        referenced as such.

        block_id: block identifier
        cl_prg: OpenCL program
        cl_ctx: OpenCL context
        """
        
        self.id = block_id
        self.ctx = cl_ctx
        self.queue = cl_queue
        
        print "\n=== block ",self.id, "==="
        
        # memory flags
        self.mf = cl.mem_flags
        
        self.mem_access = self.mf.COPY_HOST_PTR
        
        # grid data
        b = Block.blockList[block_id]
        
        if gdata.flux_method == "vanLeer":
            self.ghost = 2
        elif gdata.flux_method == "WENO5":
            self.ghost = 3
        
        # number of cells in i and j directions
        self.ni = b.nni
        self.nj = b.nnj
        
        # number of velocities
        self.Nv = gdata.Nv
        
        # boundaries for flow domain, excluding ghost cells
        self.imin = self.ghost
        self.imax = self.ni + self.ghost - 1
        self.jmin = self.ghost
        self.jmax = self.nj + self.ghost - 1
        
        # array sizes, including ghost cells
        self.Ni = self.ni + 2*self.ghost
        self.Nj = self.nj + 2*self.ghost
        
        # boundary conditions
        self.bc_list = b.bc_list
        
        # fill conditions
        self.fill_condition = b.fill_condition
        
        # x & y coordinates of the grid size = (ni + 1, nj + 1)
        self.x = b.grid.x
        self.y = b.grid.y
        
        self.centreX = np.zeros((self.ni,self.nj))
        self.centreY = np.zeros((self.ni,self.nj))
        
        # set block residual to be very high
        self.residual = 1e10
        
        ####
        # call initialising functions
        
       # generate OpenCL source
        self.block_CL()
        
        # initialise block
        self.initGeom()

        # ready to go!!
        
        print "=== block ",self.id, " initialised ===\n"
        
        return

    def block_CL(self):
        """
        generate OpenCL code from source
        """

        # PACK DICTIONARY
        data = {'ni':self.ni,'nj':self.nj, 'Ni':self.Ni, 'Nj':self.Nj,\
        'imin':self.imin, 'imax':self.imax, 'jmin':self.jmin, \
        'jmax':self.jmax, 'ghost':self.ghost, 'block_id':self.id,\
        'bc_list':self.bc_list}
        
        self.source_name = genOpenCL(data)
        f = open(self.source_name,'r')
        fstr = f.readlines()
        f.close()
        #print fstr
        
        self.source_CL = fstr
        
        self.update_CL()        
        
        return
        
    def update_CL(self):
        """
        update the CL source code
        """
        
        self.source_CL = update_source(self.source_CL, self.source_name)
        
        fstr = "".join(self.source_CL)
        
        t0 = time.clock()
        self.prg = cl.Program(self.ctx, fstr).build()
        t = time.clock() - t0
        
        print "block {}: OpenCL kernel compiled ({}s)".format(self.id, t)
        
        return
        
    def set_buffer(self,host_buffer):
        """
        initialise a buffer on the device given a pointer to a host side buffer
        """
        
        device_buffer = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mem_access, hostbuf = host_buffer)
        
        return device_buffer
        
    def set_buffer_size(self,size):
        """
        initialise a buffer on the device given the size of the buffer
        """
        
        device_buffer = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=size, hostbuf=None)
        
        return device_buffer
    
    def initGeom(self):
        
         # coordinate data, with ghost cells
        xy_H = np.ones((self.Ni+1,self.Nj+1,2),dtype=np.float64)*np.nan # x, y
        
        # initialise coordinate data
        start = np.array([self.ghost, self.ghost])
        stop = start + np.array([self.x.shape[0], self.x.shape[1]])
        
        xy_H[start[0]:stop[0],start[1]:stop[1],0] = self.x[0:(self.ni+1),0:(self.nj+1),0]
        xy_H[start[0]:stop[0],start[1]:stop[1],1] = self.y[0:(self.ni+1),0:(self.nj+1),0]
        
        # cell centres
        centre_H = -1.0*np.ones((self.Ni,self.Nj,2),dtype=np.float64) # cell centres, x, y
        
        # mid-side nodes
        side_H = -1.0*np.ones((self.Ni,self.Nj,2,2),dtype=np.float64) # mid-side node, x, y
        
        # cell area
        self.area_H = -1.0*np.ones((self.Ni,self.Nj),dtype=np.float64) # cell area
        
        # cell normals
        normal_H = -1.0*np.ones((self.Ni,self.Nj,2,2),dtype=np.float64) # side normals, x, y
        
        # cell side lengths
        length_H = -1.0*np.ones((self.Ni,self.Nj,2),dtype=np.float64) #  side length
        
        ###
        # coordinate data, with ghost cells
        self.xy_D = self.set_buffer(xy_H) # x, y
        
        # cell centres
        self.centre_D = self.set_buffer(centre_H) # cell centres, x, y
        
         # mid-side nodes
        self.side_D = self.set_buffer(side_H) # mid-side node, x, y
        
        # cell area
        self.area_D = self.set_buffer(self.area_H) # cell area
        
        # cell normals
        self.normal_D = self.set_buffer(normal_H) # side normals, x, y
        
        # cell side lengths
        self.length_D = self.set_buffer(length_H) #  side length
        
        return
        
        
    def initFunctions(self, restart_hdf=None):
        """
        initialise all functions in the flow domain of this block, excludes
        ghost cells
        """        
        
        # macro variables, without ghost cells
        #.s0 -> density
        #.s1 -> x velocity
        #.s2 -> y-velocity
        #.s3 -> 1/temperature
        
        self.macro_H = np.ones((self.ni,self.nj,4),dtype=np.float64)
        self.Q_H = np.ones((self.ni,self.nj,2),dtype=np.float64)
        
        if restart_hdf:
            step = restart_hdf['global_data/final_step'][()]
            data = restart_hdf['step_%d/block_%d'%(step, self.id)]
            self.macro_H[:,:,0] = data['rho'][()]
            self.macro_H[:,:,1:3] = data['UV'][()]
            self.macro_H[:,:,3] = 1.0/data['T'][()]
            self.Q_H[:] = data['Q'][()]
            f_H = data['f'][()]
            
        else:
            self.macro_H[:,:,0] *= self.fill_condition.D # density
            self.macro_H[:,:,1] *= self.fill_condition.U # x-velocity
            self.macro_H[:,:,2] *= self.fill_condition.V # y-velocity
            self.macro_H[:,:,3] *= 1.0/self.fill_condition.T # temperature
            self.Q_H[:,:,0] *= self.fill_condition.qx # qx
            self.Q_H[:,:,1] *= self.fill_condition.qy # qy
            
            if self.fill_condition.isUDF:
                for i in range(self.ni):
                    for j in range(self.nj):
                        x = self.centreX[i,j]
                        y = self.centreY[i,j]
                        
                        if self.fill_condition.UDF_D:
                            fval = eval(self.fill_condition.UDF_D)
                            self.macro_H[i,j,0] = fval
                        if self.fill_condition.UDF_U:
                            fval = eval(self.fill_condition.UDF_U)
                            self.macro_H[i,j,1] = fval
                        if self.fill_condition.UDF_V:
                            fval = eval(self.fill_condition.UDF_V)
                            self.macro_H[i,j,2] = fval
                        if self.fill_condition.UDF_T:
                            fval = eval(self.fill_condition.UDF_T)
                            self.macro_H[i,j,3] = 1.0/fval
                        if self.fill_condition.UDF_qx:
                            fval = eval(self.fill_condition.UDF_qx)
                            self.Q_H[i,j,0] = fval*5.0/(4 + gdata.K)
                        if self.fill_condition.UDF_qy:
                            fval = eval(self.fill_condition.UDF_qy)
                            self.Q_H[i,j,1] = fval*5.0/(4 + gdata.K)
                        
                        # hack to remove error
                        x += x
                        y += y
        # time step
        self.time_step_H = np.zeros((self.ni,self.nj),dtype=np.float64)

        ####################
        ## DEVICE SIDE DATA
        
        self.flag_H = np.zeros((2), dtype=np.int32)
        self.flag_D = self.set_buffer(self.flag_H)
        
        ## common

        f64_size = np.dtype(np.float64).itemsize
        dist_size = self.Ni*self.Nj*self.Nv*2*f64_size
        macro_size = self.Ni*self.Nj*4*f64_size

        if restart_hdf:
            self.f_D = self.set_buffer(f_H)
        else:
            self.f_D = self.set_buffer_size(dist_size)
            
        self.flux_f_S_D = self.set_buffer_size(dist_size)
        self.flux_macro_S_D = self.set_buffer_size(macro_size)
        
        self.flux_f_W_D = self.set_buffer_size(dist_size)
        self.flux_macro_W_D = self.set_buffer_size(macro_size)
        
        self.prim_D = self.set_buffer_size(macro_size)
        self.aL_D = self.set_buffer_size(macro_size)
        self.aR_D = self.set_buffer_size(macro_size)
        self.aT_D = self.set_buffer_size(macro_size)
        self.faceQ_D = self.set_buffer_size(self.Ni*self.Nj*2*f64_size)
        self.Mxi_D = self.set_buffer_size(self.Ni*self.Nj*3*f64_size)
        
        # wall properties
        wall_len = max(self.ni, self.nj)
        self.wall_prop_D = self.set_buffer_size(wall_len*4*4*f64_size) # [D,U,V,T]
        
        self.wall_cover_H = np.zeros((wall_len,4,4),dtype=np.float64)
        self.wall_cover_H[:,:,0] = gdata.vartheta_initial
        self.wall_cover_D = self.set_buffer(self.wall_cover_H) # wall coverage fraction
        
        self.wall_dist_D = self.set_buffer_size(wall_len*self.Nv*2*f64_size)
        
        self.sigma_D = self.set_buffer_size(dist_size)
        
        # macroscopic properties, without ghost cells
        self.macro_D = self.set_buffer(self.macro_H)
        self.Q_D = self.set_buffer(self.Q_H)
        
        # time step
        self.time_step_D = self.set_buffer(self.time_step_H)
        
        # create an pyopencl.array instance, this is used for reduction (max) later
        self.time_step_array = cl_array.Array(self.queue, shape=self.time_step_H.shape, dtype=np.float64, data=self.time_step_D)
            
        ### RESIDUAL
        self.residual_H = np.zeros((self.ni,self.nj,4),dtype=np.float64)
        self.residual_D = self.set_buffer(self.residual_H)
        
        ### CHECK
        self.nan_check_H = np.zeros((2),dtype=np.int32)
        self.nan_check_D = self.set_buffer(self.nan_check_H)
        
        ### INTERNAL DATA
        if gdata.save_options.internal_data:
            self.Txyz_H = np.ones((self.ni,self.nj,4),dtype=np.float64)
            self.Txyz_D = self.set_buffer(self.Txyz_H)
            
        cl.enqueue_barrier(self.queue)
        
        if not restart_hdf:
            # perform initialisation of distribution functions
            global_size, work_size = m_tuple((self.ni, self.nj, 1),(1,1,gdata.CL_local_size))
            self.prg.initFunctions(self.queue, global_size, work_size,
                                   self.f_D, self.macro_D, self.Q_D)
            
            cl.enqueue_barrier(self.queue)
            
            self.prg.calcMacro(self.queue, global_size, work_size,
                   self.f_D, self.macro_D)
                   
            cl.enqueue_barrier(self.queue)
            
            self.prg.calcQ(self.queue, global_size, work_size,
                            self.f_D, self.macro_D, self.Q_D)
                            
       
        # also update the boundary conditions
        self.parametricBC()
        
        
        print("global buffers initialised") 


    def ghostExchange(self, this_face, other_block, other_face):
        """
        perform ghost cell updating

        transfer data from "other" block to this blocks ghost cells        
        
        NOTE: assuming orientation flag is always zero (correct layout of grids)
        """
        
        faceA = np.int32(this_face)
        faceB = np.int32(other_face)

        # turn other_block index into a pointer to an object
        other_block = self.blockList[other_block]        
        
        NiB = np.int32(other_block.Ni)
        NjB = np.int32(other_block.Nj)
        
        this_f = self.f_D
        that_f = other_block.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
            
        global_size, work_size = m_tuple(global_size,(1,1,gdata.CL_local_size))
        
        self.prg.edgeExchange(self.queue, global_size, work_size,
                               this_f, self.xy_D, faceA,
                               that_f, other_block.xy_D, NiB, NjB, faceB)

    def ghostExtrapolate(self, this_face):
        """
        update the ghost cells to give constant gradient across the face
        """
        
        face = np.int32(this_face)
        
        f = self.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
        
        global_size, work_size = m_tuple(global_size,(1,1,gdata.CL_local_size))
        
        self.prg.edgeExtrapolate(self.queue, global_size, work_size,
                               f, face)
        
    def ghostInflow(self, this_face):
        """
        update the ghost distribution function with an equilibrium distribution
        defined by a passed in density and temperature and an extrapolated velocity
        normal to the wall
        """
        
        face = np.int32(this_face)
        
        f = self.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
        
        global_size, work_size = m_tuple(global_size,(1,1,gdata.CL_local_size))
        
        self.prg.edgeInflow(self.queue, global_size, work_size,
                               f, face, self.normal_D, self.macro_D, 
                               self.Q_D, self.wall_prop_D)
        
        return
        
    def ghostOutflow(self, this_face):
        """
        update the ghost distribution function with an equilibrium distribution
        defined by a passed in pressure and an extrapolated velocity
        normal to the wall. Adjust the density according to the extrapolated 
        temperature
        """
        
        face = np.int32(this_face)
        
        f = self.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
        
        global_size, work_size = m_tuple(global_size,(1,1,gdata.CL_local_size))
        
        self.prg.edgeOutflow(self.queue, global_size, work_size,
                               f, face, self.normal_D, self.macro_D, 
                               self.Q_D, self.wall_prop_D)
        
        return
        
    def ghostConstant(self, this_face):
        """
        generate distribution function values from the defined 
         constants and populate the ghost cells with this data
        """
        
        face = np.int32(this_face)
        
        f = self.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
            
        global_size, work_size = m_tuple(global_size,(1,1,gdata.CL_local_size))
        
        self.prg.edgeConstant(self.queue, global_size, work_size,
                               f, face, self.wall_prop_D)
    
    def ghostMirror(self, this_face):
        """
        update the ghost cells to give zero gradient across the face
        """
        
        face = np.int32(this_face)
        
        f = self.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
            
        global_size, work_size = m_tuple(global_size,(1,1,gdata.CL_local_size))
        
        self.prg.edgeMirror(self.queue, global_size, work_size, f, face)
                               
    def edgeConstGrad(self, this_face):
        """
        update the ghost cells to give constant gradient across the face
        """
        
        face = np.int32(this_face)
        
        f = self.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
            
        global_size, work_size = m_tuple(global_size,(1,1,gdata.CL_local_size))
        
        self.prg.edgeConstGrad(self.queue, global_size, work_size,
                               f, face)
    
    def parametricBC(self):
        """
        update the parametric boundary conditions
        """
        
        global_size, work_size = m_tuple((max(self.ni,self.nj),1),(1,1))
        
        t = np.float64(gdata.time)
        
        self.prg.paraBC(self.queue, global_size, work_size, self.para_D,
                        self.wall_prop_D, t)
        
        cl.enqueue_barrier(self.queue)
        
        return
    
    def updateBC(self):
        """
        update all boundary conditions
        this function must be called prior to every time step
        """
        
        for this_face, bc in enumerate(self.bc_list):
            
            if bc.type_of_BC in [ADJACENT, PERIODIC]:
                # exchange ghost cell information with adjacent cell
                self.ghostExchange(this_face, bc.other_block, bc.other_face)
                
            elif bc.type_of_BC == EXTRAPOLATE_OUT:
                # extrapolate the cell data to give ZERO gradient
                self.ghostExtrapolate(this_face)
            
            elif bc.type_of_BC == CONSTANT:
                # generate distribution function values from the defined 
                # constants and populate the ghost cells with this data
                self.ghostConstant(this_face)
            
            elif bc.type_of_BC == REFLECT:
                # populate ghost cells with mirror image of interior data
                """ NOTE: This only works for cartesian grids where the velocity space
                is aligned with the grid"""
                self.ghostMirror(this_face)
            
            elif bc.type_of_BC in [DIFFUSE, ADSORBING]:
                # extrapolate the cell data to give CONSTANT gradient
                self.edgeConstGrad(this_face)
                self.has_accommodating = True
                
            elif bc.type_of_BC == INFLOW:
                self.ghostInflow(this_face)
            
            elif bc.type_of_BC == OUTFLOW:
                self.ghostOutflow(this_face)
            
            #print "block {}, face {}: b.c. updated".format(self.id, this_face)            
    
    def ghostXYExchange(self, this_face, other_block, other_face):
        """
        perform ghost cell grid coordinates updating

        transfer data from "other" block to this blocks ghost cells        
        
        NOTE: assuming orientation flag is always zero (correct layout of grids)
        """
        
        faceA = np.int32(this_face)
        faceB = np.int32(other_face)

        # turn other_block index into a pointer to an object
        other_block = self.blockList[other_block]        
        
        NiB = np.int32(other_block.Ni)
        NjB = np.int32(other_block.Nj)
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost-1)
        else:
            global_size = (self.ghost-1, self.nj)
        
        self.prg.xyExchange(self.queue, global_size, None,
                               self.xy_D, faceA,
                               other_block.xy_D,
                               NiB, NjB, faceB)
    
    def ghostXYExtrapolate(self, this_face):
        """
        update the ghost cells coordinates to give zero gradient across the face
        """
        
        face = np.int32(this_face)
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost)
        else:
            global_size = (self.ghost, self.nj)
        
        self.prg.xyExtrapolate(self.queue, global_size, None,
                               self.xy_D, face)
            
    def updateGeom(self):
        """
        update all the geometric data of the cells in this block
        """
        
        ## update all x y coordinates of the ghost cells
        this_face = 0
        
        for bc in self.bc_list:       
            
            if bc.type_of_BC == ADJACENT:
                # exchange ghost cell information with adjacent cell
                self.ghostXYExchange(this_face, bc.other_block, bc.other_face)
                
            else:
                # maintain edge cell spacing and propogate as needed
                self.ghostXYExtrapolate(this_face)
            
            print "block {}, face {}: ghost coords. updated".format(self.id, this_face)
            
            this_face += 1
        
        
        # once all vertex positions are updated, we can caclulate geometric 
        #  properties of the cells
        
        self.prg.cellGeom(self.queue, (self.Ni, self.Nj), None,
                          self.xy_D, self.area_D, self.centre_D,
                          self.side_D, self.normal_D,
                          self.length_D)
        
        cl.enqueue_barrier(self.queue)
        
        # get it back for later use
        centre_H = np.ones((self.Ni,self.Nj,2),dtype=np.float64)
        length_H = np.ones((self.Ni,self.Nj,2),dtype=np.float64)
        
        cl.enqueue_copy(self.queue, self.area_H, self.area_D)
        cl.enqueue_copy(self.queue, centre_H, self.centre_D)
        cl.enqueue_copy(self.queue, length_H, self.length_D)
        
        grabI = range(self.ghost, self.Ni-self.ghost)
        grabJ = range(self.ghost, self.Nj-self.ghost)
        
        for i in range(self.ni):
            I = grabI[i]
            for j in range(self.nj):
                J = grabJ[j]
                self.centreX[i,j] = centre_H[I,J,0]
                self.centreY[i,j] = centre_H[I,J,1]
                
        # the total area of this block
        self.total_area = np.sum(self.area_H[self.ghost:-self.ghost, self.ghost:-self.ghost])
        
        # the parametric value along the side of the block for each edge cell
        para = np.zeros((max(self.ni, self.nj), 4), dtype=np.float64)
        
        lengths_north = length_H[self.ghost:(-self.ghost),-self.ghost,0]
        lengths_south = length_H[self.ghost:(-self.ghost),self.ghost,0]
        lengths_east = length_H[-self.ghost,self.ghost:(-self.ghost),1]
        lengths_west = length_H[self.ghost,self.ghost:(-self.ghost),1]
        
        self.face_lengths = [lengths_north, lengths_east, lengths_south, lengths_west]
        
        #NORTH
        stencil = lengths_north
        total_length = np.sum(stencil)
        length = 0.0
        for i in range(len(stencil)):
            cell_length = stencil[i]
            para[i,0] = (length + cell_length/2.0)/total_length
            length += cell_length      
            
        #SOUTH
        stencil = lengths_south
        total_length = np.sum(stencil)
        length = 0.0
        for i in range(len(stencil)):
            cell_length = stencil[i]
            para[i,2] = (length + cell_length/2.0)/total_length
            length += cell_length     
            
        #EAST
        stencil = lengths_east
        total_length = np.sum(stencil)
        length = 0.0
        for i in range(len(stencil)):
            cell_length = stencil[i]
            para[i,1] = (length + cell_length/2.0)/total_length
            length += cell_length   
            
        #WEST
        stencil = lengths_west
        total_length = np.sum(stencil)
        length = 0.0
        for i in range(len(stencil)):
            cell_length = stencil[i]
            para[i,3] = (length + cell_length/2.0)/total_length
            length += cell_length   
        
        para_H = para
        self.para_D = self.set_buffer(para_H)
        
        return
        
    def get_flag(self):
        """
        read the flag array back from the device
        """
        
        cl.enqueue_copy(self.queue,self.flag_H,self.flag_D)
        
        if self.flag_H[0] != 0:
            print "flag = ", self.flag_H
        
        return self.flag_H[0]
        
#===============================================================================
#   Unified Gas Kinetic Scheme (UGKS)
#===============================================================================
    
    def UGKS_flux(self):
        """
        get the fluxes for a specified distribution
        """
        south = np.int32(0)
        west = np.int32(1)
        
        dt = np.float64(gdata.dt)
        
        #--------------------------------------------------------------------
        # do the south faces of each cell
        #--------------------------------------------------------------------
        
        ##
        # calculate the interface distributions        
        ##
        global_size, work_size = m_tuple((self.ni, self.nj+1, 1), (1,1,gdata.CL_local_size))        
        self.prg.iFace(self.queue, global_size, work_size,
                   self.f_D, self.flux_f_S_D, self.sigma_D, self.centre_D,
                   self.side_D, self.normal_D, south)
                   
        cl.enqueue_barrier(self.queue)
        
        ##
        # calculate the fluxes due to accommodating walls, also determine the
        # domain that is to be used for the internal flux calculation to 
        # follow 
        ##
        offset_top = 0; offset_bot = 0
        global_size, work_size = size_cl((self.ni, 1, 1), (1, 1, gdata.CL_local_size))
        bc_type = self.bc_list[0].type_of_BC
        if bc_type in [DIFFUSE, ADSORBING]:
            offset_top = 1
            north_wall = np.int32(0)
            #print "north"
            if bc_type == DIFFUSE:
                self.prg.accommodatingWallDist(self.queue, global_size, work_size,
                                 self.normal_D, north_wall, self.wall_prop_D, 
                                 self.flux_f_S_D, dt)
            
            else:
                self.prg.adsorbingWall_P1(self.queue, global_size, work_size,
                                 self.normal_D, north_wall, self.wall_prop_D,
                                 self.wall_cover_D, self.wall_dist_D,
                                 self.flux_f_S_D, self.macro_D, dt)
                                 
                cl.enqueue_barrier(self.queue)
                
                rtype = self.bc_list[0].reflect_type
            
                if rtype == 'CL':
                    self.prg.adsorbingWallCL_P2(self.queue, global_size, work_size,
                                     self.normal_D, north_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_S_D, self.macro_D, dt)
                                     
                else:
                    self.prg.adsorbingWallDS_P2(self.queue, global_size, work_size,
                                     self.normal_D, north_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_S_D, self.macro_D, dt)
                
                cl.enqueue_barrier(self.queue)
            
            self.prg.wallFlux(self.queue, global_size, work_size,
                                 self.normal_D, self.length_D, 
                                 north_wall, self.flux_f_S_D, 
                                 self.flux_macro_S_D, dt)
            
         
        bc_type = self.bc_list[2].type_of_BC
        if bc_type in [DIFFUSE, ADSORBING]:
            offset_bot = 1
            south_wall = np.int32(2)
            #print "south"
            if bc_type == DIFFUSE:
                self.prg.accommodatingWallDist(self.queue, global_size, work_size,
                               self.normal_D, south_wall, self.wall_prop_D, 
                               self.flux_f_S_D, dt)
            
            else:
                self.prg.adsorbingWall_P1(self.queue, global_size, work_size,
                                 self.normal_D, south_wall, self.wall_prop_D,
                                 self.wall_cover_D, self.wall_dist_D,
                                 self.flux_f_S_D, self.macro_D, dt)
                                 
                cl.enqueue_barrier(self.queue)
                
                rtype = self.bc_list[2].reflect_type
                
                if rtype == 'CL':
                    self.prg.adsorbingWallCL_P2(self.queue, global_size, work_size,
                                     self.normal_D, south_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_S_D, self.macro_D, dt)
                
                else:
                    self.prg.adsorbingWallDS_P2(self.queue, global_size, work_size,
                                     self.normal_D, south_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_S_D, self.macro_D, dt)
                
            cl.enqueue_barrier(self.queue)
            
            self.prg.wallFlux(self.queue, global_size, work_size,
                               self.normal_D, self.length_D, 
                               south_wall, self.flux_f_S_D, 
                               self.flux_macro_S_D, dt)
                               
        
        ##
        # calculate the internal fluxes    
        ##
        
        shrink = offset_top + offset_bot
        offset_bot = np.int32(offset_bot)
        offset_top = np.int32(offset_top)
        
        global_size, work_size = m_tuple((self.ni, self.nj+1-shrink, 1), (1,1,gdata.CL_local_size))  
        
        self.prg.getFaceCons(self.queue, global_size, work_size,
                           self.flux_f_S_D, self.normal_D, south, 
                           self.prim_D, offset_bot, offset_top)
    
        self.prg.getAL(self.queue, global_size, work_size,
                           self.f_D, self.normal_D, south, self.aL_D, 
                           offset_bot, offset_top)
                           
        self.prg.getAR(self.queue, global_size, work_size,
                           self.f_D, self.normal_D, south,
                           self.aR_D, offset_bot, offset_top)
        
        cl.enqueue_barrier(self.queue)
        
        global_size, work_size = size_cl((self.ni, self.nj+1-shrink), (gdata.work_size_i,gdata.work_size_j))
        
        self.prg.getPLR(self.queue, global_size, work_size,
                        self.centre_D, self.side_D, south, self.prim_D, 
                        self.aL_D, self.aR_D, offset_bot, offset_top)
                           
        cl.enqueue_barrier(self.queue)
        
        self.prg.initMacroFlux(self.queue, global_size, work_size,
                               self.flux_macro_S_D, south, dt, self.prim_D,
                               self.aL_D, self.aR_D, self.aT_D, self.Mxi_D,
                               offset_bot, offset_top)
                               
        cl.enqueue_barrier(self.queue)   
                
        
        global_size, work_size = m_tuple((self.ni, self.nj+1-shrink, 1), (1,1,gdata.CL_local_size))                    
        self.prg.calcFaceQ(self.queue, global_size, work_size,
                           self.flux_f_S_D, self.prim_D, self.normal_D,
                           self.faceQ_D, south, offset_bot, offset_top)
                           
                           
        cl.enqueue_barrier(self.queue)
        
        self.prg.macroFlux(self.queue, global_size, work_size,
                           self.flux_f_S_D, self.sigma_D, self.flux_macro_S_D,
                           self.normal_D, self.length_D, south, dt, 
                           self.prim_D, self.faceQ_D, offset_bot, offset_top)
        
        self.prg.distFlux(self.queue, global_size, work_size,
                          self.flux_f_S_D, self.sigma_D, self.normal_D, 
                          self.length_D, south, dt, self.prim_D, self.aL_D,
                          self.aR_D, self.aT_D, self.Mxi_D, self.faceQ_D, 
                          offset_bot, offset_top)        
        
        cl.enqueue_barrier(self.queue)
                         
         
        #--------------------------------------------------------------------                
        # do the west faces of each cell
        #--------------------------------------------------------------------
        
        ##
        # calculate the interface distributions        
        ##

        cl.enqueue_barrier(self.queue) # ensure no overlap due to sigma_D        
        
        global_size, work_size = m_tuple((self.ni+1, self.nj, 1), (1,1,gdata.CL_local_size))        
        self.prg.iFace(self.queue, global_size, work_size,
                   self.f_D, self.flux_f_W_D, self.sigma_D, self.centre_D,
                   self.side_D, self.normal_D, west)
                   
        cl.enqueue_barrier(self.queue)
        
        ##
        # calculate the fluxes due to accommodating walls, also determine the
        # domain that is to be used for the internal flux calculation to 
        # follow 
        ##
        offset_top = 0; offset_bot = 0
        global_size, work_size = size_cl((1, self.nj, 1), (1,1,gdata.CL_local_size))
        bc_type = self.bc_list[1].type_of_BC
        if bc_type in [DIFFUSE, ADSORBING]:
            offset_top = 1
            east_wall = np.int32(1)
            #print "east"
            if bc_type == DIFFUSE:
                self.prg.accommodatingWallDist(self.queue, global_size, work_size,
                               self.normal_D, east_wall, self.wall_prop_D, 
                               self.flux_f_W_D, dt)
            
            else:
                self.prg.adsorbingWall_P1(self.queue, global_size, work_size,
                                 self.normal_D, east_wall, self.wall_prop_D,
                                 self.wall_cover_D, self.wall_dist_D,
                                 self.flux_f_W_D, self.macro_D, dt)
                                 
                cl.enqueue_barrier(self.queue)
                
                rtype = self.bc_list[1].reflect_type
                
                if rtype == 'CL':
                    self.prg.adsorbingWallCL_P2(self.queue, global_size, work_size,
                                     self.normal_D, east_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_W_D, self.macro_D, dt)
                                     
                else:                
                    self.prg.adsorbingWallDS_P2(self.queue, global_size, work_size,
                                     self.normal_D, east_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_W_D, self.macro_D, dt)
                
            cl.enqueue_barrier(self.queue)
            
            self.prg.wallFlux(self.queue, global_size, work_size,
                               self.normal_D, self.length_D, 
                               east_wall, self.flux_f_W_D, 
                               self.flux_macro_W_D, dt)
            
            
        bc_type = self.bc_list[3].type_of_BC
        if bc_type in [DIFFUSE, ADSORBING]:
            offset_bot = 1
            west_wall = np.int32(3)
            #print "west"
            if bc_type == DIFFUSE:
                self.prg.accommodatingWallDist(self.queue, global_size, work_size,
                               self.normal_D, west_wall, self.wall_prop_D, 
                               self.flux_f_W_D, dt)
           
            else:
                self.prg.adsorbingWall_P1(self.queue, global_size, work_size,
                                 self.normal_D, west_wall, self.wall_prop_D,
                                 self.wall_cover_D, self.wall_dist_D,
                                 self.flux_f_W_D, self.macro_D, dt)
                                 
                cl.enqueue_barrier(self.queue)
            
                rtype = self.bc_list[1].reflect_type
                
                if rtype == 'CL':
                                     
                    self.prg.adsorbingWallCL_P2(self.queue, global_size, work_size,
                                     self.normal_D, west_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_W_D, self.macro_D, dt)
                
                else:
                    self.prg.adsorbingWallDS_P2(self.queue, global_size, work_size,
                                     self.normal_D, west_wall, self.wall_prop_D,
                                     self.wall_cover_D, self.wall_dist_D,
                                     self.flux_f_W_D, self.macro_D, dt)
                
            cl.enqueue_barrier(self.queue)
            
            self.prg.wallFlux(self.queue, global_size, work_size,
                               self.normal_D, self.length_D, 
                               west_wall, self.flux_f_W_D, 
                               self.flux_macro_W_D, dt)
            
        ##
        # calculate the internal fluxes    
        ##
        
        shrink = offset_top + offset_bot
        offset_bot = np.int32(offset_bot)
        offset_top = np.int32(offset_top)
        
        global_size, work_size = m_tuple((self.ni+1-shrink, self.nj, 1), (1,1,gdata.CL_local_size))
        
        self.prg.getFaceCons(self.queue, global_size, work_size,
                           self.flux_f_W_D, self.normal_D, west, 
                           self.prim_D, offset_bot, offset_top)
                           
        self.prg.getAL(self.queue, global_size, work_size,
                           self.f_D, self.normal_D, west, self.aL_D, 
                           offset_bot, offset_top)
                           
        self.prg.getAR(self.queue, global_size, work_size,
                           self.f_D, self.normal_D, west,
                           self.aR_D, offset_bot, offset_top)
        
        cl.enqueue_barrier(self.queue)
        
        global_size, work_size = size_cl((self.ni+1-shrink, self.nj), (gdata.work_size_i,gdata.work_size_j))
        
        self.prg.getPLR(self.queue, global_size, work_size,
                        self.centre_D, self.side_D, west, self.prim_D, 
                        self.aL_D, self.aR_D, offset_bot, offset_top)
                        
        cl.enqueue_barrier(self.queue)
        
        self.prg.initMacroFlux(self.queue, global_size, work_size,
                               self.flux_macro_W_D, west, dt, self.prim_D,
                               self.aL_D, self.aR_D, self.aT_D, self.Mxi_D,
                               offset_bot, offset_top)
                               
        cl.enqueue_barrier(self.queue)
       
        global_size, work_size = m_tuple((self.ni+1-shrink, self.nj, 1), (1,1,gdata.CL_local_size))
        
        self.prg.calcFaceQ(self.queue, global_size, work_size,
                           self.flux_f_W_D, self.prim_D, self.normal_D,
                           self.faceQ_D, west, offset_bot, offset_top)
                           
        cl.enqueue_barrier(self.queue)

        self.prg.macroFlux(self.queue, global_size, work_size,
                           self.flux_f_W_D, self.sigma_D, self.flux_macro_W_D,
                           self.normal_D, self.length_D, west, dt, 
                           self.prim_D, self.faceQ_D, offset_bot, offset_top)
        
        self.prg.distFlux(self.queue, global_size, work_size,
                          self.flux_f_W_D, self.sigma_D, self.normal_D, 
                          self.length_D, west, dt, self.prim_D, self.aL_D,
                          self.aR_D, self.aT_D, self.Mxi_D, self.faceQ_D, 
                          offset_bot, offset_top)
                         
            
        return
        
    def UGKS_update(self, get_residual=False):
        """
        update the cell average values
        """
        
        # get the current Q value
        
        
        global_size, work_size = m_tuple((self.ni, self.nj, 1),(1,1,gdata.CL_local_size))
        self.prg.calcQ(self.queue, global_size, work_size,
                   self.f_D, self.macro_D, self.Q_D)
                   
        cl.enqueue_barrier(self.queue)
        
        # update the macro buffer
        global_size, work_size = size_cl((self.ni, self.nj),(gdata.work_size_i, gdata.work_size_j))
        self.prg.updateMacro(self.queue, global_size, work_size,
                             self.flux_macro_S_D, self.flux_macro_W_D, self.area_D,
                             self.macro_D, self.residual_D)
                             
        cl.enqueue_barrier(self.queue)                    
        
        self.nan_check_H[:] = 0
        cl.enqueue_copy(self.queue, self.nan_check_D, self.nan_check_H)
        
        dt = np.float64(gdata.dt)
        
        global_size, work_size = m_tuple((self.ni, self.nj, 1),(1,1,gdata.CL_local_size))
        self.prg.UGKS_update(self.queue, global_size, work_size,
                             self.f_D, self.flux_f_S_D, self.flux_f_W_D,
                             self.area_D, self.macro_D, self.Q_D, 
                             self.residual_D, dt, self.nan_check_D)
                             
        cl.enqueue_barrier(self.queue)
        cl.enqueue_copy(self.queue, self.nan_check_H, self.nan_check_D)
        
        if self.nan_check_H[0] == 1:
            raise RuntimeError("NaN encountered in update")
                             
        
        
        self.host_update = 0
        self.macro_update = 0
        
        if get_residual:
            
            cl.enqueue_barrier(self.queue)
            
            global_size, work_size = m_tuple((self.ni, self.nj), (gdata.work_size_i, gdata.work_size_j))        
            self.prg.getResidual(self.queue, global_size, work_size,
                             self.macro_D, self.residual_D)
                             
            cl.enqueue_barrier(self.queue)
            
            cl.enqueue_copy(self.queue,self.residual_H,self.residual_D)
            self.updateHostMacro()
            
            sum_res = np.sum(np.sum(self.residual_H, axis=0),axis=0)
            sum_avg = np.sum(np.sum(np.abs(self.macro_H), axis=0),axis=0)
            
            self.residual = np.sqrt(self.ni*self.nj*sum_res)/(sum_avg+sys.float_info.min)
        
        return
        
#==============================================================================
# INETRNAL DATA
#==============================================================================
    def getInternal(self):
        """
        get internal data from the simulation
        """
        global_size = (self.ni, self.nj)
        
        if gdata.save_options.internal_data:
            global_size, work_size = m_tuple((self.ni, self.nj),(gdata.work_size_i,gdata.work_size_j))
            self.prg.getInternalTemp(self.queue, global_size, work_size,
                                 self.f_D, self.Txyz_D)
            cl.enqueue_barrier(self.queue)
                                 
            cl.enqueue_copy(self.queue,self.Txyz_H,self.Txyz_D)
        return

    def updateHostMacro(self):
        """
        grab macro from device
        """
        
        if self.macro_update == 0:
            cl.enqueue_barrier(self.queue)
            cl.enqueue_copy(self.queue,self.macro_H,self.macro_D)
            cl.enqueue_copy(self.queue,self.wall_cover_H,self.wall_cover_D)
            self.macro_update = 1
            
        return
    
    def updateHost(self, getF = False):
        """
        update host data arrays
        """
        
        if self.host_update == 0:
            cl.enqueue_barrier(self.queue)
            
            # grab the primary variables
            self.updateHostMacro()
            
            # have to calculate the heat flux vector
            global_size, work_size = m_tuple((self.ni, self.nj, 1),(1,1,gdata.CL_local_size))
            
            self.prg.calcQ(self.queue, global_size, work_size,
                   self.f_D, self.macro_D, self.Q_D)
            cl.enqueue_barrier(self.queue)
            
            cl.enqueue_copy(self.queue,self.Q_H,self.Q_D)
            
            # adjust the heat flux from internal to external
            self.Q_H *= (4 + gdata.K)/5.0
            
            self.host_update = 1
        
            # get internal data, if specified
            self.getInternal()
            
        if getF:
            f_H = np.ones((self.Ni,self.Nj,self.Nv,2),dtype=np.float64)
            cl.enqueue_copy(self.queue,f_H,self.f_D)
            return f_H
            
        return
    
    def getDT(self):
        """
        return the minimum time step allowable for this block
        """
        
        self.prg.clFindDT(self.queue, (self.ni, self.nj), None,
                          self.xy_D, self.area_D, self.macro_D, self.time_step_D)
                          
        cl.enqueue_barrier(self.queue)
        
        # run reduction kernel
        max_freq = cl_array.max(self.time_step_array,queue=self.queue).get()
        
        
        return gdata.CFL/max_freq
        
    def save_hdf(self, h5Name, grp, step, all_data=False):
        """
        save block data to hdf_file
        """
        xdmf = ""
        
        sgrp = grp.require_group("block_" + str(self.id))
        
        sgrp.create_dataset("dt",data=gdata.dt)
        sgrp.create_dataset("time",data=gdata.time)
        
        if all_data:
            f_H = self.updateHost(getF = all_data)
        else:
            self.updateHost(getF = all_data)
        
        sgrp.create_dataset("rho",data=self.macro_H[:,:,0], compression=gdata.save_options.compression)
        xdmf += '<Attribute Name="rho" AttributeType="Scalar" Center="Cell">\n'
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/rho\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '</Attribute>\n'
        
        sgrp.create_dataset("UV",data=self.macro_H[:,:,1:3], compression=gdata.save_options.compression)
        # fancy shenanigans to get a zero valued third element in the vector
        xdmf += '<Attribute Name="UV" AttributeType="Vector" Center="Cell">\n'
        xdmf += '<DataItem Dimensions="%d %d 3" Function="JOIN($0, $1)" ItemType="Function">\n'%(self.ni, self.nj)
        xdmf += '<DataItem Dimensions="%d %d 2" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/UV\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '<DataItem Dimensions="%d %d 1" Function="ABS($0 - $0)" ItemType="Function">\n'%(self.ni, self.nj)
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/rho\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '</DataItem>\n'
        xdmf += '</DataItem>\n'
        xdmf += '</Attribute>\n'
        
        sgrp.create_dataset("Q",data=self.Q_H, compression=gdata.save_options.compression)
        xdmf += '<Attribute Name="Q" AttributeType="Vector" Center="Cell">\n'
        xdmf += '<DataItem Dimensions="%d %d 3" Function="JOIN($0, $1)" ItemType="Function">\n'%(self.ni, self.nj)
        xdmf += '<DataItem Dimensions="%d %d 2" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/Q\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '<DataItem Dimensions="%d %d 1" Function="ABS($0 - $0)" ItemType="Function">\n'%(self.ni, self.nj)
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/rho\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '</DataItem>\n'
        xdmf += '</DataItem>\n'
        xdmf += '</Attribute>\n'
        
        
        if gdata.save_options.internal_data:
            self.getInternal()
            sgrp.create_dataset("Txyz",data=self.Txyz_H[:,:,0:-1], compression=gdata.save_options.compression)
            
            xdmf += '<Attribute Name="Txyz" AttributeType="Vector" Center="Cell">\n'
            xdmf += '<DataItem Dimensions="%d %d 3" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
            xdmf += '%s:/step_%d/block_%d/Txyz\n'%(h5Name, step, self.id)
            xdmf += '</DataItem>\n'
            xdmf += '</Attribute>\n'

        sgrp.create_dataset("T",data=1.0/self.macro_H[:,:,3], compression=gdata.save_options.compression)
        
        xdmf += '<Attribute Name="T" AttributeType="Scalar" Center="Cell">\n'
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/T\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '</Attribute>\n'
        xdmf += '<Attribute Name="P" AttributeType="Scalar" Center="Cell">\n'
        xdmf += '<DataItem Dimensions="%d %d" Function="$0*$1/2.0" ItemType="Function">\n'%(self.ni, self.nj)
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/rho\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/T\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '</DataItem>\n'
        xdmf += '</Attribute>\n'
        

        ads = sgrp.require_group("adsorbed")
        cover_N = self.wall_cover_H[0:self.ni,0,0]
        cover_E = self.wall_cover_H[0:self.nj,1,0]
        cover_S = self.wall_cover_H[0:self.ni,2,0]
        cover_W = self.wall_cover_H[0:self.nj,3,0]
        ads.create_dataset("cover_N",data=cover_N, compression=gdata.save_options.compression)
        ads.create_dataset("cover_E",data=cover_E, compression=gdata.save_options.compression)
        ads.create_dataset("cover_S",data=cover_S, compression=gdata.save_options.compression)
        ads.create_dataset("cover_W",data=cover_W, compression=gdata.save_options.compression)
        
        reflected_N = self.wall_cover_H[0:self.ni,0,1]
        reflected_E = self.wall_cover_H[0:self.nj,1,1]
        reflected_S = self.wall_cover_H[0:self.ni,2,1]
        reflected_W = self.wall_cover_H[0:self.nj,3,1]
        ads.create_dataset("reflected_N",data=reflected_N, compression=gdata.save_options.compression)
        ads.create_dataset("reflected_E",data=reflected_E, compression=gdata.save_options.compression)
        ads.create_dataset("reflected_S",data=reflected_S, compression=gdata.save_options.compression)
        ads.create_dataset("reflected_W",data=reflected_W, compression=gdata.save_options.compression)
        
        adsorbed_N = self.wall_cover_H[0:self.ni,0,2]
        adsorbed_E = self.wall_cover_H[0:self.nj,1,2]
        adsorbed_S = self.wall_cover_H[0:self.ni,2,2]
        adsorbed_W = self.wall_cover_H[0:self.nj,3,2]
        ads.create_dataset("adsorbed_N",data=adsorbed_N, compression=gdata.save_options.compression)
        ads.create_dataset("adsorbed_E",data=adsorbed_E, compression=gdata.save_options.compression)
        ads.create_dataset("adsorbed_S",data=adsorbed_S, compression=gdata.save_options.compression)
        ads.create_dataset("adsorbed_W",data=adsorbed_W, compression=gdata.save_options.compression)
        
        desorbed_N = self.wall_cover_H[0:self.ni,0,3]
        desorbed_E = self.wall_cover_H[0:self.nj,1,3]
        desorbed_S = self.wall_cover_H[0:self.ni,2,3]
        desorbed_W = self.wall_cover_H[0:self.nj,3,3]
        ads.create_dataset("desorbed_N",data=desorbed_N, compression=gdata.save_options.compression)
        ads.create_dataset("desorbed_E",data=desorbed_E, compression=gdata.save_options.compression)
        ads.create_dataset("desorbed_S",data=desorbed_S, compression=gdata.save_options.compression)
        ads.create_dataset("desorbed_W",data=desorbed_W, compression=gdata.save_options.compression)     
        
            
        if all_data:
            sgrp = grp.require_group("block_" + str(self.id))

            sgrp.create_dataset("f",data=f_H, compression=gdata.save_options.compression)
                
        
        return xdmf
        
        
    def block_mass(self):
        """
        calculate the total mass in this block
        """
        
        self.updateHost()
        
        
        
        # mass in the bulk of the flow
        bulk_mass = np.sum(self.area_H[self.ghost:-self.ghost,self.ghost:-self.ghost]*self.macro_H[:,:,0])
        bulk_mass *= gdata.D_ref
        
        
        # mass adsorbed on the wall
        adsorbed_mass = 0.0
        cover_N = self.wall_cover_H[0:self.ni,0,0]
        cover_E = self.wall_cover_H[0:self.nj,1,0]
        cover_S = self.wall_cover_H[0:self.ni,2,0]
        cover_W = self.wall_cover_H[0:self.nj,3,0]
        cover = [cover_N, cover_E, cover_S, cover_W]
        for face in range(4):
           adsorbed_mass += np.sum(cover[face]*gdata.S_T*self.face_lengths[face]*gdata.L_ref)
        
#        print "bulk mass = ",bulk_mass
#        print "adsorbed mass = ",adsorbed_mass
        
        return bulk_mass, adsorbed_mass