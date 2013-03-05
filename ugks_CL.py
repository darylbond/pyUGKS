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
        s += '#pragma OPENCL EXTENSION cl_amd_fp64 : enable \n'
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
    s += '#define LEFT 0\n'
    s += '#define RIGHT 1\n'
    s += '#define FLUX_IN_N 1.0\n'
    s += '#define FLUX_IN_E 1.0\n'
    s += '#define FLUX_IN_S -1.0\n'
    s += '#define FLUX_IN_W -1.0\n'
    s += '#define BLOCK {}\n'.format(data['block_id'])
    s += '#define FLUX_METHOD {}\n'.format(flux_method)
    s += '#define STENCIL_LENGTH {}\n'.format(stencil_length)
    s += '#define MID_STENCIL {}\n'.format(mid_stencil)
    s += '\n'
    
    s += '#define PI %0.15e\n'%pi
    s += '#define SPI %0.15e\n'%sqrt(pi)
    s += '#define S2P %0.15e\n\n'%sqrt(2.0*pi)
    
    s += '#define Pr {}\n'.format(gdata.Pr)
    s += '#define Cs {}\n'.format(sqrt(gdata.gamma))
    s += '#define umax {}\n'.format(gdata.umax)
    s += '#define vmax {}\n'.format(gdata.vmax)
    s += '#define Kn {}\n'.format(gdata.Kn)
    s += '#define gam {}\n'.format(gdata.gamma)
    s += '#define B {}\n'.format(gdata.b)
    s += '#define K {} // total number of extra degrees of freedom accounted for\n'.format(gdata.K)
    s += '#define chi {}\n\n'.format(gdata.chi)
    
    s += '#define CFL %0.15e\n\n'%gdata.CFL
    
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
    has_diffuse = 0
    diffuse_list = []
    for bc in bc_list:
        if bc.type_of_BC == DIFFUSE:
            val = 1
            has_diffuse = 1
        else:
            val = 0
        s += str(val)
        diffuse_list.append(val)
        count += 1
        if count < 4:
            s += ', '
    s += '}; // flag for diffuse boundary condition\n'
    
    s += '#define DIFFUSE_NORTH {}\n'.format(diffuse_list[0])
    s += '#define DIFFUSE_EAST {}\n'.format(diffuse_list[1])
    s += '#define DIFFUSE_SOUTH {}\n'.format(diffuse_list[2])
    s += '#define DIFFUSE_WEST {}\n'.format(diffuse_list[3])
    s += '#define HAS_DIFFUSE_WALL {}\n'.format(has_diffuse)
    
    s += '__constant double4 BC_cond[4] = {'
    count = 0
    for bc in bc_list:
        if count > 0:
            s += '                                 '
        if bc.type_of_BC == DIFFUSE:
            s += '(double4)('
            s += '1.0, %g, %g, %g),\n'%(bc.Uwall, bc.Vwall, 1.0/bc.Twall)
        else:
            s += '(double4)('
            s += '0.0, 0.0, 0.0, 0.0),\n'
        count += 1
    s = s.rstrip(',\n')
    s += '}; // if flagged for diffuse BC, use these values for (D, U, V, 1/T)\n'
    
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
    