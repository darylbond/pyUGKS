# pyUGKS.py
# Daryl Bond

#==============================================================================
# IMPORT
#==============================================================================

# global

import os

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

#local

from ugks_data import *
from ugks_sim import UGKSim

#==============================================================================
# CLASS
#==============================================================================

class UGKS(object):
    def __init__(self, jobName, svg=False, vtk=False, add_str=""):
        """
        UGKS object
        """
        
        # set to default values
        gdata.startup() 
        
        # Step 1: Initialise the global data
        print "\n LOAD SIMULATION FILE\n"
        global_preparation(jobName, add_str)

        self.sim = UGKSim()
        
        if svg and gdata.dimensions == 2:
            gdata.sketch.write_svg_file(gdata, FlowCondition.flowList,
            Block.blockList, faceList2D)

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