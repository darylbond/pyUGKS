# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:26:45 2011

@author: uqdbond1
"""

# codeCL.py
import os
from math import pi
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

#import source.source_CL as sl
from ugks_data import gdata
from geom.geom_bc_defs import *
from geom.geom_defs import *

def genHeader(data):
    """
    create all header stuff - output as string
    """
    
    ## LOCAL VARIABLES
    quad = gdata.quad
    weight = gdata.weight
    
    bc_list = data['bc_list']
    
    s = ''
        
    if gdata.flux_method == "vanLeer":
        flux_method = 0
        stencil_length = 3
        mid_stencil = 1
    elif gdata.flux_method == "WENO5":
        flux_method = 1
        stencil_length = 5
        mid_stencil = 2
    
    if gdata.platform == "AMD":
        #s += '#pragma OPENCL EXTENSION cl_amd_fp64 : enable \n'
        s += '#pragma OPENCL EXTENSION cl_amd_printf : enable\n\n'
    elif gdata.platform == "Intel":
        s += '#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n'
        #s += '#pragma OPENCL EXTENSION cl_intel_printf : enable\n\n'
    elif gdata.platform == "NVIDIA":
        s += '#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n\n'
    
    s += '/////////////////////////////////////////\n'
    s += '//CONSTANTS \n'
    s += '/////////////////////////////////////////\n\n'
    s += '#define LOCAL_SIZE %d\n'%gdata.CL_local_size
    s += '#define LOCAL_LOOP_LENGTH %d\n'%(np.ceil(gdata.Nv/float(gdata.CL_local_size)))
    s += '#define NI {}\n'.format(data['Ni'])
    s += '#define NJ {}\n'.format(data['Nj'])
    s += '#define ni {}\n'.format(data['ni'])
    s += '#define nj {}\n'.format(data['nj'])
    s += '#define NV {}\n'.format(gdata.Nv)
    s += '#define NX {}\n'.format(data['Ni']+1)
    s += '#define NY {}\n'.format(data['Nj']+1)
    s += '#define IMIN {}\n'.format(data['imin'])
    s += '#define IMAX {}\n'.format(data['imax'])
    s += '#define JMIN {}\n'.format(data['jmin'])
    s += '#define JMAX {}\n'.format(data['jmax'])
    s += '#define GHOST {}\n'.format(data['ghost'])
    s += '#define SOUTH 0\n'
    s += '#define WEST 1\n'
    s += '#define GNORTH 0\n'
    s += '#define GEAST 1\n'
    s += '#define GSOUTH 2\n'
    s += '#define GWEST 3\n'
    s += '#define BLOCK {}\n'.format(data['block_id'])
    s += '#define FLUX_METHOD {}\n'.format(flux_method)
    s += '#define STENCIL_LENGTH {}\n'.format(stencil_length)
    s += '#define MID_STENCIL {}\n'.format(mid_stencil)
    s += '#define RELAX_TYPE {}\n'.format(gdata.relax_type)
    s += '\n'
    
    s += '#define PI %0.15e\n'%pi
    
    s += '#define Pr {}\n'.format(gdata.Pr)
    s += '#define umax {}\n'.format(gdata.umax)
    s += '#define vmax {}\n'.format(gdata.vmax)
    s += '#define Kn {}\n'.format(gdata.Kn)
    s += '#define gam {}\n'.format(gdata.gamma)
    s += '#define B {}\n'.format(gdata.b)
    s += '#define K {} // total number of extra degrees of freedom accounted for\n'.format(gdata.K)
    s += '#define chi {}\n\n'.format(gdata.chi)
    
    s += '#define CFL %0.15e\n\n'%gdata.CFL
    
#==============================================================================
#     # boundary factors
#==============================================================================
    # put in the adsorbing boundary stuff regardless of if it is needed
    s += '__constant double BETA_N[4] = {'
    for bc in bc_list:
        if bc.type_of_BC == ADSORBING:
            s += '%0.15e, '%bc.beta_n
        else: s += '-1, '
    s = s[:-2] # remove comma and space
    s += '};\n'
    
    s += '__constant double BETA_T[4] = {'
    for bc in bc_list:
        if bc.type_of_BC == ADSORBING:
            s += '%0.15e, '%bc.beta_t
        else: s += '-1, '
    s = s[:-2] # remove comma and space
    s += '};\n'
    
    s += '__constant double GAMMA_F[4] = {'
    for bc in bc_list:
        if bc.type_of_BC == ADSORBING:
            s += '%0.15e, '%bc.gamma_f
        else: s += '-1, '
    s = s[:-2] # remove comma and space
    s += '};\n'
    
    s += '__constant double ALPHA_P[4] = {'
    for bc in bc_list:
        if bc.type_of_BC == ADSORBING:
            s += '%0.15e, '%bc.alpha_p
        else: s += '-1, '
    s = s[:-2] # remove comma and space
    s += '};\n'
    
    s += '__constant double ALPHA_N[4] = {'
    for bc in bc_list:
        if bc.type_of_BC == ADSORBING:
            s += '%0.15e, '%bc.alpha_n
        else: s += '-1, '
    s = s[:-2] # remove comma and space
    s += '};\n'
            
    s += '__constant double ALPHA_T[4] = {'
    for bc in bc_list:
        if bc.type_of_BC == ADSORBING:
            s += '%0.15e, '%bc.alpha_t
        else: s += '-1, '
    s = s[:-2] # remove comma and space
    s += '};\n'
    
    # the number of points in the look-up table
#    s += '__constant int N_ISO[4] = {'
#    for bc in bc_list:
#        if bc.type_of_BC == ADSORBING:
#            s += '%d, '%bc.adsorb.shape[0]
#        else: s += '-1, '
#    s = s[:-2] # remove comma and space
#    s += '};\n'
    
    # define the look-up tables for adsorption isotherms
    
    for bci, bc in enumerate(bc_list):
        if bc.type_of_BC == ADSORBING:
            shape = bc.adsorb.shape
            if shape[1] != 3:
                raise RuntimeError("Passed in array for isotherm values is not an n*3 array")
            s += '__constant double4 ISO_%s[%d] = {'%(faceName[bci], shape[0])
            for i in range(shape[0]):
                s += '(double4)('
                for j in range(shape[1]):
                    s += '%0.15e, '%bc.adsorb[i,j]
                s = s[:-2]
                s += ', 0), '
            s = s[:-2] # remove comma and space
            s += '};\n'
        else: 
            s += '__constant double4 ISO_%s[1] = {(double4)(-1,-1,-1,-1)};\n'%(faceName[bci])
    
    plotting = 0
    
    if plotting:
        fig = plt.figure()
     
    n_tri = []
    for bci, bc in enumerate(bc_list):
        if bc.type_of_BC == ADSORBING:
            
            xy = bc.adsorb[:,0:2]
            deln = Delaunay(xy)
            tris = deln.vertices
            nbrs = deln.neighbors
            
            if plotting:
                ax = fig.add_subplot(1,len(bc_list),bci+1)
                cb = ax.tricontourf(xy[:,0], xy[:,1], tris, bc.adsorb[:,2],50)
                ax.triplot(xy[:,0], xy[:,1], tris)
                plt.colorbar(cb)

            shape = tris.shape

            s += '__constant int4 TRI_%s[%d] = {'%(faceName[bci], shape[0])
            for i in range(shape[0]):
                s += '(int4)('
                for j in range(shape[1]):
                    s += '%d, '%tris[i,j]
                s = s[:-2]
                s += ', 0), '
            s = s[:-2] # remove comma and space
            s += '};\n'
        
            n_tri.append(shape[0])
            
            shape = nbrs.shape

            s += '__constant int4 NBR_%s[%d] = {'%(faceName[bci], shape[0])
            for i in range(shape[0]):
                s += '(int4)('
                for j in range(shape[1]):
                    s += '%d, '%nbrs[i,j]
                s = s[:-2]
                s += ', 0), '
            s = s[:-2] # remove comma and space
            s += '};\n'
            
        else: 
            s += '__constant int4 TRI_%s[1] = {(int4)(-1,-1,-1,-1)};\n'%(faceName[bci])
            s += '__constant int4 NBR_%s[1] = {(int4)(-1,-1,-1,-1)};\n'%(faceName[bci])
            n_tri.append(0)
    
    # the number of points in the look-up table
    s += '__constant int N_TRI[4] = {%d, %d, %d, %d};\n'%(n_tri[0],n_tri[1],n_tri[2],n_tri[3])
    
    if plotting:
        plt.show()
    
    
        
    has_diffuse = False
    has_CL = False
    has_S = False
    has_D = False
    for bc in bc_list:
        if bc.type_of_BC == DIFFUSE:
            has_diffuse = True            
        elif bc.type_of_BC == ADSORBING:
            if bc.reflect_type == 'S':
                has_S = True
            elif bc.reflect_type == 'D':
                has_D = True
            elif bc.reflect_type == 'CL':
                has_CL = True

    s += '#define HAS_DIFFUSE_WALL %d\n'%has_diffuse
    s += '#define HAS_ADSORBING_CL_WALL %d\n'%has_CL
    s += '#define HAS_ADSORBING_SPECULAR_WALL %d\n'%has_S
    s += '#define HAS_ADSORBING_DIFFUSE_WALL %d\n'%has_D

    wall_name = ['N','E','S','W']    
    
    for i, bc in enumerate(bc_list):
        if bc.type_of_BC in [ADSORBING, DIFFUSE, CONSTANT, INFLOW, OUTFLOW]:
            if bc.UDF_D:
                st = bc.UDF_D
            else:
                st = str(bc.D)
            s += '#define WALL_%s_D %s\n'%(wall_name[i],st)
            if bc.UDF_U:
                st = bc.UDF_U
            else:
                st = str(bc.U)
            s += '#define WALL_%s_U %s\n'%(wall_name[i],st)
            if bc.UDF_V:
                st = bc.UDF_V
            else:
                st = str(bc.V)
            s += '#define WALL_%s_V %s\n'%(wall_name[i],st)
            if bc.UDF_T:
                st = bc.UDF_T
            else:
                st = str(bc.T)
            s += '#define WALL_%s_T 1.0/(%s)\n'%(wall_name[i],st)
            
            if bc.type_of_BC == OUTFLOW:
                s += '#define WALL_%s_P %s\n'%(wall_name[i],bc.P)
            else:
                s += '#define WALL_%s_P -1\n'%wall_name[i]
                
        else:
            s += '#define WALL_%s_D -1\n'%wall_name[i]
            s += '#define WALL_%s_U -1\n'%wall_name[i]
            s += '#define WALL_%s_V -1\n'%wall_name[i]
            s += '#define WALL_%s_T -1\n'%wall_name[i]
            s += '#define WALL_%s_P -1\n'%wall_name[i]
            
#==============================================================================
#     
#==============================================================================
        
    
    
    s += '__constant double2 QUAD[{}] = {{'.format(gdata.Nv)
    
    for i in range(gdata.Nv):
        if i > 0:
            s += '                              '
        if i < gdata.Nv-1:
            s += '(double2)('
            s += '%0.15e, '%quad[i,0]
            s += '%0.15e '%quad[i,1]
            s += '),\n'
        else:
            s += '(double2)('
            s += '%0.15e,'%quad[i,0]
            s += '%0.15e'%quad[i,1]
            s += ')'
    s += '};\n\n'
    
    count = 0
    s += '__constant double WEIGHT[{}] = {{'.format(gdata.Nv)
    for i in range(gdata.Nv):
        count += 1
        if i < gdata.Nv-1:
            s += '%0.15e, '%weight[i]
        else:
            s += '%0.15e'%weight[i]
        if count == 4:
            s+= '\n                              '
            count = 0
    s += '};\n\n'
    
    count = 0
    s += '__constant size_t mirror_NS[{}] = {{'.format(gdata.Nv)
    for i in range(gdata.Nv):
        count += 1
        if i < gdata.Nv-1:
            s += '{}, '.format(gdata.mirror_NS[i])
        else:
            s += '{}'.format(gdata.mirror_NS[i])
        if count == 10:
            s+= '\n                              '
            count = 0
    s += '};\n'
    
    count = 0
    s += '__constant size_t mirror_EW[{}] = {{'.format(gdata.Nv)
    for i in range(gdata.Nv):
        count += 1
        if i < gdata.Nv-1:
            s += '{}, '.format(gdata.mirror_EW[i])
        else:
            s += '{}'.format(gdata.mirror_EW[i])
        if count == 10:
            s+= '\n                              '
            count = 0
    s += '};\n'
    
    count = 0
    s += '__constant size_t mirror_D[{}] = {{'.format(gdata.Nv)
    for i in range(gdata.Nv):
        count += 1
        if i < gdata.Nv-1:
            s += '{}, '.format(gdata.mirror_D[i])
        else:
            s += '{}'.format(gdata.mirror_D[i])
        if count == 10:
            s+= '\n                              '
            count = 0
    s += '};\n\n'
            
    
    return s

def genOpenCL(data):
    """
    write all opencl code for one block to a text file with the number of 
    the block
    """
    
    block_id = data['block_id']
    
    ##------------- CODE GENERATION ------------------------------------------------
    
    print " generating complete OpenCL code...",
    
    # EXTERNAL SOURCE
    src = gdata.source
    
    # HEADER
    header = genHeader(data)
    
    # CONCATENATE SOURCES

    # order of compilation
    file_list = ["definitions","clFunctions.c", "clAdsorb.c","clKernel.c",\
                'clUGKS.c',"clGhostRoutines.c"]
    
    # the string that everything is added to
    src_str = header
    
    split = "\n//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    
    for s in file_list:
        src_str += split
        src_str += src[s]
    
    print "done"
    
    # save source
    print " write OpenCL source to file...",
    # Write one file per block.
    (dirName,firstName) = os.path.split(gdata.rootName)
    sourcePath = os.path.join(gdata.rootName,"OpenCL")
    if not os.access(sourcePath, os.F_OK):
        os.makedirs(sourcePath)
    fileName = firstName+(".OpenCL.b%04d" % block_id)
    fileName = os.path.join(sourcePath, fileName)
    
    fileName += ".c"    
    
    fp = open(fileName, "w")
    fp.write(src_str)
    fp.close()
    
    print "done"
    
    return fileName

def update_source(source, fileName):
    """
    look through the code and update any #defines that need it
    """
    
    for i, line in enumerate(source):
        if "#define LOCAL_SIZE" in line:
            source[i] = '#define LOCAL_SIZE %d\n'%gdata.CL_local_size
        elif '#define LOCAL_LOOP_LENGTH' in line:
            source[i] = '#define LOCAL_LOOP_LENGTH %d\n'%(np.ceil(gdata.Nv/float(gdata.CL_local_size)))
            break
        
    fp = open(fileName, "w")
    fp.write("".join(source))
    fp.close()
    
    return source