# pyUGKS.py
# Daryl Bond

#==============================================================================
# IMPORT
#==============================================================================

# global

import os

import matplotlib
matplotlib.use('Agg')

os.environ["PYOPENCL_COMPILER_OUTPUT"]="1"
os.environ["PYOPENCL_NO_CACHE"]="1"

#local

from ugks_data import *
from ugks_sim import UGKSim

#==============================================================================
# CLASS
#==============================================================================

class UGKS(object):
    def __init__(self, jobName, svg=False, svg_only=False, vtk=False, additional=""):
        """
        UGKS object
        """
        
        # set to default values
        gdata.startup() 
        
        # Step 1: Initialise the global data
        print "\n LOAD SIMULATION FILE\n"
        global_preparation(jobName, additional)
        
        if svg:
            gdata.sketch.write_svg_file(gdata, FlowCondition.flowList,
            Block.blockList, faceList2D)

        if not svg_only:
            self.sim = UGKSim()

        return

    def run(self, jobName=""):
        """
        calls all scripts as required to run a simulation as defined
         in python script input to this function.
        """

        print "\n RUNNING SIMULATION... \n"

        self.sim.run()

        print "\n FINISHED SIMULATION RUN \n"

        return