# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:26:45 2011

@author: uqdbond1
"""

# codeCL.py
import os
from math import pi, sqrt
import numpy as np

#import source.source_CL as sl
from ugks_data import gdata
from geom.geom_bc_defs import *

def genHeader(data):
    """
    create all header stuff - output as string
    """
    
    ## LOCAL VARIABLES
    quad = gdata.quad
    weight = gdata.weight
    
    bc_list = data['bc_list']
    
    print gdata.platform
    
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
    
    # boundary factors
    s += '#define BETA_N %0.15e\n'%gdata.beta_n
    s += '#define BETA_T %0.15e\n'%gdata.beta_t
    s += '#define GAMMA_F %0.15e\n'%gdata.gamma_f
    s += '#define GAMMA_B %0.15e\n'%gdata.gamma_b
    s += '#define ALPHA_P %0.15e\n'%gdata.alpha_p
    s += '#define ALPHA_N %0.15e\n'%gdata.alpha_n
    s += '#define ALPHA_T %0.15e\n'%gdata.alpha_t
    s += '#define VARTHETA_LANGMUIR %0.15e\n'%gdata.vartheta_langmuir
    s += '\n'
    
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
    s += '};\n\n'
    
    # boundary codition
    s += '__constant uint BC_flag[4] = {'
    count = 0
    has_accommodating = 0
    accommodating_list = []
    for bc in bc_list:
        if bc.type_of_BC in  [ACCOMMODATING, CONSTANT]:
            val = 1
            if gdata.boundary_type == "diffuse":
                has_accommodating = 1
            elif gdata.boundary_type == "adsorb_CL":
                has_accommodating = 2
            elif gdata.boundary_type == "adsorb_specular-diffuse":
                has_accommodating = 3
        else:
            val = 0
        s += str(val)
        accommodating_list.append(val)
        count += 1
        if count < 4:
            s += ', '
    s += '}; // flag for accommodating boundary condition\n'
    
    s += '#define HAS_ACCOMMODATING_WALL {}\n'.format(has_accommodating)

    wall_name = ['N','E','S','W']    
    
    for i, bc in enumerate(bc_list):
        if bc.type_of_BC in [ACCOMMODATING, CONSTANT, INFLOW, OUTFLOW]:
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
    file_list = ["definitions","clFunctions.c","clKernel.c",\
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