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

from ugks_data import Block, gdata
from ugks_CL import genOpenCL
from geom.geom_bc_defs import *
from geom.geom_defs import *

def m_tuple(x, y):
    """
    multiply two tuples or lists, return a tuple
    """
    return tuple(a*b for a, b in zip(x, y))

class UGKSBlock(object):
    """
    A block that contains all required information for the simulation of 
    a single block.
    """
    
    blockList = []
    
    host_update = 0 # flag to indicate if the host is up to date
    
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
        
        # work-group size
        self.work_size = (1,1,gdata.CL_local_size)
        
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
        'bc_list':self.bc_list, 'work_size':self.work_size}
        
        name = genOpenCL(data)
        f = open(name,'r')
        fstr = "".join(f.readlines())
        f.close()
        #print fstr
        
        t0 = time.clock()
        self.prg = cl.Program(self.ctx, fstr).build()
        t = time.clock() - t0
        
        print "block {}: OpenCL kernel compiled ({}s)".format(self.id, t)
        return
        
    def set_buffer(self,host_buffer):
        """
        initialise a buffer on the device
        """
        
        device_buffer = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mem_access, hostbuf = host_buffer)
        
        return device_buffer
    
    def initGeom(self):
        
         # coordinate data, with ghost cells
        self.xy_H = np.ones((self.Ni+1,self.Nj+1,2),dtype=np.float64) # x, y
        
        # initialise coordinate data
        start = np.array([self.ghost, self.ghost])
        stop = start + np.array([self.x.shape[0], self.x.shape[1]])
        
        self.xy_H[start[0]:stop[0],start[1]:stop[1],0] = self.x[0:(self.ni+1),0:(self.nj+1),0]
        self.xy_H[start[0]:stop[0],start[1]:stop[1],1] = self.y[0:(self.ni+1),0:(self.nj+1),0]
        
        # cell centres
        self.centre_H = -1.0*np.ones((self.Ni,self.Nj,2),dtype=np.float64) # cell centres, x, y
        
        # mid-side nodes
        self.side_H = -1.0*np.ones((self.Ni,self.Nj,2,2),dtype=np.float64) # mid-side node, x, y
        
        # cell area
        self.area_H = -1.0*np.ones((self.Ni,self.Nj),dtype=np.float64) # cell area
        
        # cell normals
        self.normal_H = -1.0*np.ones((self.Ni,self.Nj,2,2),dtype=np.float64) # side normals, x, y
        
        # cell side lengths
        self.length_H = -1.0*np.ones((self.Ni,self.Nj,2),dtype=np.float64) #  side length
        
        ###
        # coordinate data, with ghost cells
        self.xy_D = self.set_buffer(self.xy_H) # x, y
        
        # cell centres
        self.centre_D = self.set_buffer(self.centre_H) # cell centres, x, y
        
         # mid-side nodes
        self.side_D = self.set_buffer(self.side_H) # mid-side node, x, y
        
        # cell area
        self.area_D = self.set_buffer(self.area_H) # cell area
        
        # cell normals
        self.normal_D = self.set_buffer(self.normal_H) # side normals, x, y
        
        # cell side lengths
        self.length_D = self.set_buffer(self.length_H) #  side length
        
        return
        
        
    def initFunctions(self, restart_hdf=None):
        """
        initialise all functions in the flow domain of this block, excludes
        ghost cells
        """        
        
        ####################
        ## HOST SIDE DATA
        
        self.flag_H = np.zeros((2), dtype=np.int32)
        
        ## common
        self.f_H = -1.0*np.ones((self.Ni,self.Nj,self.Nv,2),dtype=np.float64) # mass.x, energy.y
        
        self.flux_f_H = np.zeros((self.Ni,self.Nj,self.Nv,2),dtype=np.float64) # mass.x, energy.y
        self.flux_macro_H = np.zeros((self.Ni,self.Nj,4),dtype=np.float64)

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
            self.macro_H[:,:,1:2] = data['UV'][()]
            self.macro_H[:,:,3] = 1.0/data['T'][()]
            self.Q_H[:] = data['Q'][()]
            self.f_H[:] = data['f'][()]
            
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
        # time step
        self.time_step_H = np.zeros((self.ni,self.nj),dtype=np.float64)

        ####################
        ## DEVICE SIDE DATA
        
        self.flag_D = self.set_buffer(self.flag_H)
        
        ## common
        self.f_D = self.set_buffer(self.f_H)
        
        self.flux_f_D = self.set_buffer(self.flux_f_H)
        self.flux_macro_D = self.set_buffer(self.flux_macro_H)
        
        # macroscopic properties, without ghost cells
        self.macro_D = self.set_buffer(self.macro_H)
        self.Q_D = self.set_buffer(self.Q_H)
        
        # time step
        self.time_step_D = self.set_buffer(self.time_step_H)
        
        # create an pyopencl.array instance, this is used for reduction (max) later
        self.time_step_array = cl_array.Array(self.queue, shape=self.time_step_H.shape, dtype=np.float64, data=self.time_step_D)
            
            
        ### RESIDUAL
        if gdata.residual_options.get_residual:
            self.residual_H = np.zeros((self.ni,self.nj,2),dtype=np.float64)
            self.residual_D = self.set_buffer(self.residual_H)
            self.residual_array = cl_array.Array(self.queue, shape=self.residual_H.shape, dtype=np.float64, data=self.residual_D)
            self.f_copy_H = np.zeros((self.Ni,self.Nj,self.Nv,2),dtype=np.float64)
            self.f_copy_D = self.set_buffer(self.f_copy_H)
        
        ### INTERNAL DATA
        if gdata.save_options.internal_data:
            self.Txyz_H = np.ones((self.ni,self.nj,4),dtype=np.float64)
            self.Txyz_D = self.set_buffer(self.Txyz_H)
            
        cl.enqueue_barrier(self.queue)
        
        if not restart_hdf:
            # perform initialisation of distribution functions
            global_size = m_tuple((self.ni, self.nj, 1),self.work_size)
            self.prg.initFunctions(self.queue, global_size, self.work_size,
                                   self.f_D, self.macro_D, self.Q_D)
            cl.enqueue_barrier(self.queue)
            
            global_size = (self.ni, self.nj)
            self.prg.calcMacro(self.queue, global_size, (1,1),
                   self.f_D, self.macro_D, self.Q_D)
            cl.enqueue_barrier(self.queue)
            
            #cl.enqueue_copy(self.queue, self.f_H, self.f_D)
            #print np.sum(self.f_H[:,:,:,0],axis=2)
        
        
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
            
        global_size = m_tuple(global_size,self.work_size)
        
        self.prg.edgeExchange(self.queue, global_size, self.work_size,
                               this_f, faceA,
                               that_f, NiB, NjB, faceB)

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
        
        global_size = m_tuple(global_size,self.work_size)
        
        self.prg.edgeExtrapolate(self.queue, global_size, self.work_size,
                               f, face)
        
    
    def ghostConstant(self, this_face):
        """
        generate distribution function values from the defined 
         constants and populate the ghost cells with this data
        """
        bc = self.bc_list[this_face]
        
        face = np.int32(this_face)
        
        D = np.float64(bc.Dwall)
        U = np.float64(bc.Uwall)
        V = np.float64(bc.Vwall)
        T = np.float64(bc.Twall)
        
        f = self.f_D
        
        if this_face in [NORTH, SOUTH]:
            global_size = (self.ni, self.ghost, 1)
        else:
            global_size = (self.ghost, self.nj, 1)
            
        global_size = m_tuple(global_size,self.work_size)
        
        self.prg.edgeConstant(self.queue, global_size, self.work_size,
                               f, face, D, U, V, T)
    
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
            
        global_size = m_tuple(global_size,self.work_size)
        
        self.prg.edgeMirror(self.queue, global_size, self.work_size, f, face)
                               
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
            
        global_size = m_tuple(global_size,self.work_size)
        
        self.prg.edgeConstGrad(self.queue, global_size, self.work_size,
                               f, face)
        
    def updateBC(self):
        """
        update all boundary conditions
        this function must be called prior to every time step
        """
        
        this_face = 0
        
        for bc in self.bc_list:
            
            if bc.type_of_BC == ADJACENT:
                # exchange ghost cell information with adjacent cell
                self.ghostExchange(this_face, bc.other_block, bc.other_face)
                
            if bc.type_of_BC == PERIODIC:
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
            
            elif bc.type_of_BC == DIFFUSE:
                # extrapolate the cell data to give CONSTANT gradient
                self.edgeConstGrad(this_face)
            
            #print "block {}, face {}: b.c. updated".format(self.id, this_face)            
            
            this_face += 1
    
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
            global_size = (self.ni, self.ghost)
        else:
            global_size = (self.ghost, self.nj)
        
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
        
        cl.enqueue_barrier(self.queue)
        cl.enqueue_copy(self.queue, self.xy_H, self.xy_D)
        
        
        # once all vertex positions are updated, we can caclulate geometric 
        #  properties of the cells
        
        self.prg.cellGeom(self.queue, (self.Ni, self.Nj), None,
                          self.xy_D, self.area_D, self.centre_D,
                          self.side_D, self.normal_D,
                          self.length_D)
        
        cl.enqueue_barrier(self.queue)
        
        # get it all back for later use
        cl.enqueue_copy(self.queue, self.area_H, self.area_D)
        cl.enqueue_copy(self.queue, self.centre_H, self.centre_D)
        cl.enqueue_copy(self.queue, self.side_H, self.side_D)
        cl.enqueue_copy(self.queue, self.normal_H, self.normal_D)
        cl.enqueue_copy(self.queue, self.length_H, self.length_D)
        
        grabI = range(self.ghost, self.Ni-self.ghost)
        grabJ = range(self.ghost, self.Nj-self.ghost)
        
        for i in range(self.ni):
            I = grabI[i]
            for j in range(self.nj):
                J = grabJ[j]
                self.centreX[i,j] = self.centre_H[I,J,0]
                self.centreY[i,j] = self.centre_H[I,J,1]
                
        # the total area of this block
        self.total_area = np.sum(self.area_H[self.ghost:-self.ghost, self.ghost:-self.ghost])
        
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
        
        global_size = m_tuple((self.Ni, self.Nj, 1),self.work_size)
        self.prg.zeroFluxF(self.queue, global_size, self.work_size, self.flux_f_D)
        self.prg.zeroFluxM(self.queue, (self.Ni, self.Nj), (1,1), self.flux_macro_D)
        cl.enqueue_barrier(self.queue)
        
        dt = np.float64(gdata.dt)
        
        for face in range(2):
            for even_odd in range(2):
                ni = int(np.floor((self.ni - even_odd)/2.0))
                nj = int(np.floor((self.nj - even_odd)/2.0))
                
                global_size = [(self.ni, nj+1), (ni+1, self.nj)]
                
                self.prg.UGKS_flux(self.queue, global_size[face], (1,1),
                                       self.f_D, self.flux_f_D, self.flux_macro_D,
                                       self.centre_D, self.side_D,
                                       self.normal_D, self.length_D, self.area_D,
                                       np.int32(face), np.int32(even_odd),
                                       dt).wait()
        
#        self.queue.finish()
#        #cl.enqueue_copy(self.queue,self.flux_macro_H,self.flux_macro_D)
#        cl.enqueue_copy(self.queue,self.flux_f_H,self.flux_f_D)
#        
#        print "\nFLUX BLOCK %d"%self.id
#        #print "macro 1/T: ",self.flux_macro_H[self.ghost:-self.ghost,self.ghost:-self.ghost,3]
#        print "f: ",self.flux_f_H[self.ghost:-self.ghost,self.ghost:-self.ghost,0,0]
        
        #sys.exit()
        
        return
        
    def UGKS_update(self):
        """
        update the cell average values
        """
        
#        print "\nORIGINAL BLOCK %d"%self.id
#        cl.enqueue_copy(self.queue,self.macro_H,self.macro_D)
#        cl.enqueue_copy(self.queue,self.f_H,self.f_D)
#        print "macro: ",self.macro_H[:,:,0]
#        print "f: ",self.f_H[self.ghost:-self.ghost,self.ghost:-self.ghost,0,0]
        
        dt = np.float64(gdata.dt)
        
        global_size = m_tuple((self.ni, self.nj, 1),self.work_size)
        self.prg.UGKS_update(self.queue, global_size, self.work_size,
                             self.f_D, self.flux_f_D, self.flux_macro_D, 
                             self.macro_D, dt)
                             
        cl.enqueue_barrier(self.queue)
        
#        self.queue.finish()
#        cl.enqueue_copy(self.queue,self.macro_H,self.macro_D)
#        cl.enqueue_copy(self.queue,self.f_H,self.f_D)
#        
#        print "\nUPDATE BLOCK %d"%self.id
#        print "macro: ",self.macro_H[:,:,0]
#        print "f: ",self.f_H[self.ghost:-self.ghost,self.ghost:-self.ghost,0,0]
        
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
            self.prg.getInternalTemp(self.queue, global_size, None,
                                 self.f_D, self.Txyz_D)
            cl.enqueue_barrier(self.queue)
                                 
            cl.enqueue_copy(self.queue,self.Txyz_H,self.Txyz_D)
        return
    
    def updateHost(self, getF = False):
        """
        update host data arrays
        """
        
        #print "block = {}".format(self.id)
        
        if self.host_update == 0:
            cl.enqueue_barrier(self.queue)
            cl.enqueue_copy(self.queue,self.macro_H,self.macro_D)
            cl.enqueue_copy(self.queue,self.Q_H,self.Q_D)
            
            self.host_update == 1
        
        # get internal data, if specified
        self.getInternal()
        
        if getF:
            cl.enqueue_copy(self.queue,self.f_H,self.f_D)
        
        return
    
    def getDT(self):
        """
        return the minimum time step allowable for this block
        """
        
        self.prg.clFindDT(self.queue, (self.ni, self.nj), None,
                          self.xy_D, self.area_D, self.macro_D, self.time_step_D)
                          
        cl.enqueue_barrier(self.queue)
        
        #cl.enqueue_copy(self.queue,self.time_step_H,self.time_step_D)
        #print self.time_step_H
        
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
        xdmf += '<DataItem Dimensions="%d %d" Function="$0*$1" ItemType="Function">\n'%(self.ni, self.nj)
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/rho\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(self.ni, self.nj)
        xdmf += '%s:/step_%d/block_%d/T\n'%(h5Name, step, self.id)
        xdmf += '</DataItem>\n'
        xdmf += '</DataItem>\n'
        xdmf += '</Attribute>\n'
        
            
        if all_data:
            sgrp = grp.require_group("block_" + str(self.id))

            sgrp.create_dataset("f",data=self.f_H, compression=gdata.save_options.compression)
                
        
        return xdmf
        
        
    def block_mass(self):
        """
        calculate the total mass in this block
        """
        
        self.updateHost()
        
        mass = np.sum(self.area_H[self.ghost:-self.ghost,self.ghost:-self.ghost]*self.macro_H[:,:,0])
        
        return mass