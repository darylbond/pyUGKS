# ugks_data.py

"""
ugks_data.py: Python program to specify the flow problem.

It is intended for the user to define the flow simulation in terms of
the data objects defined in this program.
As part of its initialization, geomPrep.py will execute a user-specified
job file that contains, in Python, the user's script that defines both
geometry and flow details.
"""

# ----------------------------------------------------------------------
# System
import os
import numpy as np

#local
from pygeom.pygeom import *
from geom.geom_defs import *
from geom.geom_block import * # Block defined here, all blocks in Block.blockList
from geom.geom_flow import FlowCondition  # FlowCondition defined here, all flow conditions in FlowCondition.flowList
from geom.geom_render import *
from geom.geom_render import SketchEnvironment

import source.source_CL as sl

#----------------------------------------------------------------------

class SaveOptions:
    """
    All the options to do with saving of files in here
    
    * data:
        * save: to save or not to save
        * save_name: string appended to name when saving files
        * writeHDF: True/False
        * writeVTK: True/False
        * save_initial_f: save full initial dist functions, True/False
        * save_final_f: save full final dist functions, True/False
        * internal_data: calculate translational temperatures etc
        * compression: type of compression used in hdf files
        * save_count: number of iterations between save events
    """
    
    __slots__ = 'save','save_name','writeHDF','writeVTK','save_initial_f',
    'save_final_f','internal_data','compression','save_count','h5Name'
    
    save = False
    save_name = ""
    h5Name = ""
    save_initial_f = False
    save_final_f = False
    internal_data = False
    compression = 'gzip'
    save_count = 1

#----------------------------------------------------------------------
class ResidualOptions:
    """
    All the options to do with residuals
    
    * data:
        * global_residual: current residual value
        * residual_history: history of global residual values
        * get_residual: calculate the residual, True/False
        * residual_step: how often to calculate the residual
        * min_residual: a minimum cutoff for the allowable residual
        * min_slope: minimum slope of residual on loglog plot
        * slope_start: how many entries before sampling the slope
        * slope_sample: the number of points to calculate the slope from
        * non_linear_output: output the residual with an exponential relationship in residual_count
        * only do non-linear up to *limit and then use *limit as residual_count
    """
    
    global_residual = 1e50
    residual_history = []
    get_residual = False
    residual_count = -1
    min_residual = 1e-50
    min_slope = 1e-3
    slope_start = 10
    slope_sample = 3
    non_linear_output = False
    non_linear_output_limit = 1e5
    non_linear_save = False
    plot_residual = False

#----------------------------------------------------------------------

class UGKSData(object):
    """
    Organise, remember and write the global data.
    """
    count = 0
    
    # We want to prevent the user's script from introducing new attributes
    # via typographical errors.
    __slots__ = 'dimensions', 'title',\
                't0', 'dt', 'CFL', 'time',\
                'print_count', 'max_time', 'max_step', 'dt_plot', \
                'chi', 'R', 'gamma', 'Pr', 'Kn', \
                'Nv', 'b', 'K', \
                'T_ref', 'L_ref', 'C_ref', 't_ref', 'D_ref',\
                'step','CL_local_size',\
                'quad','weight',\
                'mirror_NS',\
                'mirror_EW','dt_update_count','umax','vmax',\
                'rootName',\
                'device','flux_method', 'platform',\
                'plot_options','save_options','residual_options',\
                'sketch','source','check_err_count',\
                'exit','config_string',\
                'runtime_conf', 'restart'
    
    def __init__(self):
        """
        Accepts user-specified data and sets defaults. Make one only.
        """
        if UGKSData.count >= 1:
            raise Exception, "Already have a GlobalData object defined."
            
        return
    def startup(self):
        """
        initialise all values to defaults
        * items
            * dimensions: number of dimensions
            * exit: if true the simulation loop is interrupted, set in conf file
            * 
            * config_string: string containing all input defining the running of the program
            * runtime_conf: location of .conf file used for runtime modification of values
            
        """
        
        self.dimensions = 2
        self.exit = False
        self.device = "CPU"
        self.platform = "AMD"
        self.title = "Another ALBatroSS Simulation."
        self.rootName = ""
        self.config_string = ""
        self.runtime_conf = ""
        self.flux_method = "WENO5"
        self.CFL = 0.3
        self.t0 = 0.0
        self.time = 0.0
        self.step = 0
        
        # may be useful to change t0 if we are restarting from another job
        self.dt = 0.0
        self.print_count = 20
        self.max_time = 1.0e-3
        self.max_step = 1
        self.dt_plot = 1.0e-3
        self.dt_update_count = 1
        self.check_err_count = -1
        
        # gas data
        self.chi = 0.81  # power law constant determining relationship between temperature and viscosity
        self.R = 287.0  # gas constant, J/kgK
        self.gamma = 5.0/3.0    #ratio of specific heats
        self.Pr = 0.72      # ratio momentum to thermal diffusivity
        self.Kn = 0.0001    # Knudsen number, ratio mean free path to reference length
        
        
        # Reference Quantities -> for non-dimensionalising
        self.L_ref = 0.0        # length, m
        self.D_ref = 0.0        # density, kg/m^3
        self.T_ref = 273.0       # temperature, K
        
        # EXTERNAL SOURCE        
        src_loader = sl.SourceLoader()
        self.source = src_loader.src
        
        self.CL_local_size = 128 # the local work size for openCL
        
        # saving options
        self.save_options = SaveOptions()
        self.restart = False
        
        # residual options
        self.residual_options = ResidualOptions()
        
        # Make one instance to accumulate the settings for 2D SVG rendering.
        self.sketch = SketchEnvironment()
        
        UGKSData.count += 1
        
        return
        
    def init_quad(self):
        """
        quadrature rule
        """
        
        # for now we will just use a given velocity set
        n = 28
        self.Nv = n**2
        
        # offset the quadrature array
        u_mid = 0.0
        v_mid = 0.0
        
        vcoords = [ -0.5392407922630E+01, -0.4628038787602E+01, -0.3997895360339E+01, -0.3438309154336E+01,
                    -0.2926155234545E+01, -0.2450765117455E+01, -0.2007226518418E+01, -0.1594180474269E+01,
                    -0.1213086106429E+01, -0.8681075880846E+00, -0.5662379126244E+00, -0.3172834649517E+00,
                    -0.1331473976273E+00, -0.2574593750171E-01, +0.2574593750171E-01, +0.1331473976273E+00,
                    +0.3172834649517E+00, +0.5662379126244E+00, +0.8681075880846E+00, +0.1213086106429E+01,
                    +0.1594180474269E+01, +0.2007226518418E+01, +0.2450765117455E+01, +0.2926155234545E+01,
                    +0.3438309154336E+01, +0.3997895360339E+01, +0.4628038787602E+01, +0.5392407922630E+01 ]

        weights = [ +0.2070921821819E-12, +0.3391774320172E-09, +0.6744233894962E-07, +0.3916031412192E-05,
                    +0.9416408715712E-04, +0.1130613659204E-02, +0.7620883072174E-02, +0.3130804321888E-01,
                    +0.8355201801999E-01, +0.1528864568113E+00, +0.2012086859914E+00, +0.1976903952423E+00,
                    +0.1450007948865E+00, +0.6573088665062E-01, +0.6573088665062E-01, +0.1450007948865E+00,
                    +0.1976903952423E+00, +0.2012086859914E+00, +0.1528864568113E+00, +0.8355201801999E-01,
                    +0.3130804321888E-01, +0.7620883072174E-02, +0.1130613659204E-02, +0.9416408715712E-04,
                    +0.3916031412192E-05, +0.6744233894962E-07, +0.3391774320172E-09, +0.2070921821819E-12 ]
        
        self.quad = np.zeros((self.Nv,2))
        self.weight = np.zeros((self.Nv))
        
        index_array = np.zeros((n,n), dtype=np.int)
        count = 0
        for i in range(n):
            for j in range(n):
                index_array[i,j] = count
                u = vcoords[i] + u_mid
                v = vcoords[j] + v_mid
                self.quad[count,0] = u
                self.quad[count,1] = v
                self.weight[count] = weights[i]*np.exp(u**2)*weights[j]*np.exp(v**2)
                count += 1

        self.umax = abs(np.max(vcoords))
        self.vmax = abs(np.max(vcoords))
        
        self.mirror_NS = np.ravel(np.fliplr(index_array))
        self.mirror_EW = np.ravel(np.flipud(index_array))
        
        
        return
    

    def init_ref_values(self):
        """
        generate reference quantities for non-dimensionalising
        """
        # Reference Quantities -> for non-dimensionalising wrt reference quantities
        self.C_ref = sqrt(2*self.R*self.T_ref)    # speed, m/s
        self.t_ref = self.L_ref/self.C_ref      # time, s
        
        # internal degrees of freedom
        self.b = 2.0/(self.gamma - 1.0)     # number of dimensions present
        self.K = self.b - self.dimensions   # number of dimensions added to simulation
        
        if self.device == "CPU":
            self.CL_local_size = 1
        
        # Gauss-Hermite
        self.init_quad()
    
    def get_time(self):
        """
        return the time in reference, not lattice, space
        """
        
        return self.time
        
    def read_conf(self):
        """
        read the config file generated at initialisation
        """
        
        execfile(self.runtime_conf, globals())
        
        return
        
    def update(self):
        """
        run a check over things that may need updating, update if necessary
        """
            
        if gdata.residual_options.plot_residual:
            gdata.residual_options.get_residual = True
            
        return
        
# We will create just one UGKSData object that the user can alter.
gdata = UGKSData()
        
#===============================================================================
# FUNCTIONS
#===============================================================================

def non_dimensionalise_all():
    """
    take all input and NON-DIMENSIONALISE it in preparation for simulation
    This done to keep core simulation code as similar as possible to 
        reference works, also gives simpler equations.
    """
    # flow conditions
    for f in FlowCondition.flowList:
        f.D /= gdata.D_ref
        f.U /= gdata.C_ref
        f.V /= gdata.C_ref
        f.T /= gdata.T_ref
        f.qx /= (gdata.D_ref*gdata.C_ref**3)
        f.qy /= (gdata.D_ref*gdata.C_ref**3)
        f.tau /= gdata.t_ref
        
        # user defined functions
        if f.UDF_U:
            f.UDF_U = "("+f.UDF_U +")/(" + str(gdata.C_ref) + ")"
            print f.UDF_U
        if f.UDF_V:
            f.UDF_V = "("+f.UDF_V +")/(" + str(gdata.C_ref) + ")"
        if f.UDF_T:
            f.UDF_T = "("+f.UDF_T +")/(" + str(gdata.T_ref) + ")"
    
    #block conditions
    for b in Block.blockList:
        #boundary conditions
        for bc in b.bc_list:
            bc.Dwall /= gdata.D_ref
            bc.Twall /= gdata.T_ref
            bc.Uwall /= gdata.C_ref
            bc.Vwall /= gdata.C_ref
        #grid
        b.grid.x /= gdata.L_ref
        b.grid.y /= gdata.L_ref
        b.grid.z /= gdata.L_ref
    
    # global data items
    gdata.t0 /= gdata.t_ref
    gdata.dt /= gdata.t_ref
    gdata.max_time /= gdata.t_ref
    gdata.dt_plot /= gdata.t_ref
    
    return

def write_grid_files(blockList):
    print "Begin write grid file(s)."
    # Grid already created in main loop
    # Write one file per block.
    (dirName,firstName) = os.path.split(gdata.rootName)
    gridPath = os.path.join(gdata.rootName,"grid", "t0000")
    if not os.access(gridPath, os.F_OK):
        os.makedirs(gridPath)
    for b in blockList:
        fileName = firstName+(".grid.b%04d.t0000" % b.blkId)
        fileName = os.path.join(gridPath, fileName)
#        if zipFiles:
#            fp = GzipFile(fileName+".gz", "wb")
#        else:
        fp = open(fileName+".vts", "w")
        b.grid.write_block_in_VTK_format(fp)
        fp.close()
    print "End write grid file(s)."
    return

def global_preparation(jobName="", jobString=""):
    """
    prepare the domain for simulation
    get user input and define flow domain
    """
    print "jobName = ",jobName
    jobName = os.getcwd()+'/'+jobName
    rootName, ext = os.path.splitext(jobName)
    jobPath, fileName = os.path.split(rootName)
    #print "rootName = ",rootName
    
    gdata.rootName = rootName
    
    # make folder
    if not os.access(rootName, os.F_OK):
        os.makedirs(rootName)
    
    
    if ext:
        jobFileName = jobName
    else:
        jobFileName = rootName + ".py"
        
    # if extra strinbgs have been passed in, evaluate them first
    #  may still be over-written
    if jobString:
        exec(jobString,globals())
    
    # The user-specified input comes in the form of Python code.
    # In a parallel calculation, all processes should see the same setup.
    execfile(jobFileName, globals())
    
    # create config file
    conf_name = os.path.join(rootName,fileName+'.conf')
    f_conf = open(conf_name,'w')
    f_conf.write("gdata.exit = False")
    f_conf.close()
    
    gdata.runtime_conf = conf_name
    
    fstr = jobString
    
    f = open(jobFileName,'r')
    f_lines = f.readlines()
    f.close()
    
    fstr_2 = "".join(f_lines)
    
    gdata.config_string = fstr + '\n\n' + fstr_2
    
    
    
    if len(Block.blockList) < 1:
        print "Warning: no blocks defined."
    
    
        
    # svg location
    gdata.sketch.root_file_name = os.path.join(rootName,fileName)
    
    # generate reference values
    gdata.init_ref_values()

    # non-dimensionalise ALL values used in simulation
    non_dimensionalise_all()

    return
