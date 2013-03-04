#ugks_sim.py

#system
import pyopencl as cl
import numpy as np

import matplotlib.pylab as plt

from tvtk.api import tvtk
import time
import os
#from math import sqrt
import h5py

#local
from ugks_data import gdata
from ugks_data import Block
from ugks_block import UGKSBlock

class UGKSim(object):
    """
    class that handles actual simulation
    """

    step = 0
    t = 0.0

    def __init__(self):
        """
        initialise class
        """

        ## OpenCL initialisation
        print "\n INITIALISE OPENCL\n"
        self.init_CL()
        
        ## restart??
        self.restart()

        ## Blocks
        print "\n LOAD BLOCKS\n"
        self.load_blocks()

        # connections
        print "\n SET BLOCK CONNECTIONS \n"
        self.connections = Block.connectionList

        # initialise all geometric data
        print "\n INITIALISE GEOMETRY\n"
        self.initGeom()

        # initialise all distributions
        print "\n INITIALISE DISTRIBUTION FUNCTIONS\n"
        self.initFunctions()
        
        print "\n INITIALISE TIME STEP\n"
        self.updateTimeStep()

        print "\n INITIALISATION COMPLETE\n"
        
        self.time_history_residual = []
        self.time_history_residual_N = []
                
        
        if self.restart_hdf:
            self.restart_hdf.close()

        # flags
        self.saved = False
        self.closed = False
        self.HDF_init = False
        
        # save to file
        self.saveToFile(save_f=gdata.save_options.save_initial_f)

        return

    def init_CL(self):
        """
        initialise the OpenCL context
        """

        # memory flags
        self.mf = cl.mem_flags

#===============================================================================
#         # primary select device
#===============================================================================

        if gdata.platform == "AMD":
            platform = 'AMD Accelerated Parallel Processing'
        elif gdata.platform == "Intel":
            platform = 'Intel(R) OpenCL'
        elif gdata.platform == "NVIDIA":
            platform = "NVIDIA CUDA"

        self.platform = platform

        for found_platform in cl.get_platforms():
            if found_platform.name == platform:
                my_platform = found_platform
                print "OpenCL Platform = ",my_platform.name

        devices = []
        for found_device in my_platform.get_devices():
            if found_device.type == eval('cl.device_type.'+gdata.device):
                devices.append(found_device)

        if len(devices) == 1:
            device = devices[0]
        else:
            print "Available devices:"
            i = 0
            for d in devices:
                print '%d : '%i,
                print d.name
                i += 1

            select = raw_input('Which device? Input a number between 0 and '+str(len(devices)-1)+':')
            device = devices[int(select)]

        print "OpenCL device = ",device.name

        # general context
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        return

    def load_blocks(self):
        """
        load blocks into list
        """
        self.blocks = []

        self.Nb = len(Block.blockList)  # number of blocks
        for i in range(self.Nb):
            block = UGKSBlock(i, self.ctx, self.queue)
            self.blocks.append(block)

        # give a pointer to the list of simBlocks to each simBlock instance
        for b in self.blocks:
            b.blockList = self.blocks

        return

    def initGeom(self):
        """
        update the coordinates of the ghost cells of all blocks
        """

        for b in self.blocks:
            b.updateGeom()

    def initFunctions(self):
        """
        update the initial distributions
        """

        for b in self.blocks:
            b.initFunctions(self.restart_hdf)

        return
        
    def updateTimeStep(self):
        """
        update the global time step
        """
        
        dt = []

        for b in self.blocks:
            block_dt = b.getDT()
            dt.append(block_dt)

        print "dt = %g -->"%gdata.dt,
        gdata.dt = min(dt)
        print "dt = %g"%gdata.dt

        return

    def updateAllBC(self):
        """
        update the ghost cells of all blocks
        """

        for b in self.blocks:
            b.updateBC()

        return

    def one_step(self, get_dt, get_res, check_err=False):
        """
        run one step of the simulation
        """
        
        if get_res:
            res = 0.0

        # make sure all ghost cells are up to date
        self.updateAllBC()
        
        cl.enqueue_barrier(self.queue)

        for b in self.blocks:
            b.UGKS_flux()
            cl.enqueue_barrier(self.queue)
            b.UGKS_update(get_res)
            
        cl.enqueue_barrier(self.queue)
        
        if get_dt:
            self.updateTimeStep()

        if get_res:
            for b in self.blocks:
                block_res = b.residual
                res += block_res
            res /= len(self.blocks)

        # update time counter - lattice units
        gdata.time += gdata.dt

        if (gdata.max_time - gdata.time) < gdata.dt:
            gdata.dt = gdata.max_time - gdata.time

        if get_res:
            gdata.residual_options.global_residual = res
            gdata.residual_options.residual_history.append(res)
        
        
        self.step += 1
        gdata.step = self.step
        
        self.queue.flush()

        return cl.enqueue_barrier(self.queue)
        
    def run(self):
        """
        run the simulation
        """
        
        print "\n\n RUNNING SIM\n"
        
        gdata.update()
        save = gdata.save_options
        res = gdata.residual_options
        
        # plotting
        if res.plot_residual:
            self.plotResidualInit()
            
        mag = 1
        
        # error checking
        foundNaN = False

        t0 = time.time()
        
        self.step = gdata.step
        
        # save files?
        if save.save:
            self.saveToFile(save_f=save.save_initial_f)
        
            
        while self.step < gdata.max_step:
            
            ## FLAGS
            if res.get_residual:
                if (res.non_linear_output & (self.step%mag == 0)) & (self.step < res.non_linear_output_limit):
                    mag = int(10**(np.floor(np.log10(self.step+1))))
                    res.residual_count = mag
                    if res.non_linear_save:
                        save.save_count = mag
            
            if res.get_residual & (self.step%res.residual_count == 0):
                get_res = True
            else:
                get_res = False
            
            if gdata.dt_update_count < 0:
                if self.step == -gdata.dt_update_count:
                    get_dt = True
                else:
                    get_dt = False
            elif not self.step % gdata.dt_update_count:
                get_dt = True
            else:
                get_dt = False
                
            if (not self.step % gdata.check_err_count) & (gdata.check_err_count != -1):
                print "error check"
                check_err = True
            else:
                check_err = False
                
            # one iteration of the method
            step_finished = self.one_step(get_dt, get_res, check_err)
            
            self.saved = False

            gdata.read_conf()
                    
            ###
            # CONDITIONALS
            ###
            
            if gdata.exit:
                print "simulation exit -> interrupted"
                print "step ",self.step," t = ",gdata.get_time()
                break
            
            # exit condition: sim has generated a NaN
            if foundNaN:
                print "simulation exit -> NaN encountered"
                print "step ",self.step," t = ",gdata.get_time()
                break
            
            # exit condition maximum time reached
            if gdata.time >= gdata.max_time:
                print "simulation exit -> time limit reached"
                print "step ",self.step," t = ",gdata.get_time()
                break
            
            if get_res:
                self.time_history_residual.append(res.global_residual)
                self.time_history_residual_N.append(self.step)
                if res.plot_residual:
                    self.plotResidualUpdate(self.step)
                if np.any(res.global_residual <= res.min_residual):
                    print "simulation exit -> minimum residual reached"
                    print "step ",self.step," t = ",gdata.get_time()
                    break
                if (len(self.time_history_residual) >= res.slope_sample) & (self.step >= res.slope_start):
                    # fit a line to the last three data points
                    slope = np.polyfit(np.log(self.time_history_residual_N[-3::]), np.log(self.time_history_residual[-3::]), 1)
                    slope = np.abs(slope[0])
                    print "residual slope = ",slope
                    if np.any(slope < res.min_slope):
                        print "simulation exit -> residual stabilised"
                        print "step ",self.step," t = ",gdata.get_time()
                        break
            
            # print progress
            if ((not self.step % gdata.print_count) & (gdata.print_count > 0)) | get_res:
                step_finished.wait()
                print "step %d, t = %0.5f"%(self.step, gdata.get_time())
                if not res.get_residual:
                    sec_per_step = (time.time()-t0)/float(self.step)
                    steps_remaining_time = (gdata.max_time - gdata.time)/gdata.dt
                    steps_remaining_steps = gdata.max_step - self.step
                    steps_remaining = min(steps_remaining_time,steps_remaining_steps)
                    time_remaining = steps_remaining*sec_per_step
                
                    print " --> t_final-t_now = %s"%self.secToTime(time_remaining)
            
            # save to file
            if not self.step % save.save_count:
                step_finished.wait()
                if self.step == gdata.max_step:
                    self.saveToFile(save_f=save.save_final_f)
                else:
                    self.saveToFile()

        self.queue.finish()        
        
        ##
        # MAIN LOOP END
        ##
        
        if self.step >= gdata.max_step:
            print "simulation exit -> step count reached"
            print " -> step ",self.step," t = ",gdata.get_time()
        
        print "wall clock time of simulation run = ",self.secToTime(time.time()-t0)
        
        self.saveToFile(close_file=True, save_f=save.save_final_f)
        
        

        if res.plot_residual:
            plt.ioff()
            print "close residual plot to end run"
            plt.show()
                
        print " \n\nFINISHED\n"
        
        return


    def updateHost(self, getF = False):
        """
        update host data arrays
        """
        for b in self.blocks:
            b.updateHost(getF)

        return
        
    def plotResidualInit(self):
        """
        plot time history of residual
        """
        
        if not gdata.residual_options.plot_residual:
            return

        # generate plot
        self.resFig = plt.figure(figsize=(12, 6))

        #turn on interactive
        
        self.resPlot = self.resFig.add_subplot(111)
        
        self.line_residual_0, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual,'r',label="rho")
        self.line_residual_1, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual, 'g', label="U")
        self.line_residual_2, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual, 'b', label="V")
        self.line_residual_3, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual, 'k', label="1/T")
        
        plt.legend(loc=3)        
        
        if not self.time_history_residual:
            max_res = 1
        else:
            max_res = np.max(self.time_history_residual[-1])
        
        plot_title = "Residual"
        if gdata.title:
            plot_title += " - " + gdata.title
        plt.title(plot_title)
        
        plt.grid(True)
        self.residual_plot_limits = True
        plt.xlim(1,gdata.max_step)
        plt.ylim(gdata.residual_options.min_residual,max_res)
        plt.ion()
        plt.show()
        

        # throw away first few samples
        self.res_throw = True

        return

    def plotResidualUpdate(self, fit=False):
        """
        update the residual plot
        """
        
        if not self.resPlot:
            return
        #print "plotting residual...",
        
        #self.resPlot.hold(False)
        #self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual)
        
#        if self.residual_plot_limits:
#            plt.ylim(gdata.residual_options.min_residual,self.time_history_residual[0])
#            self.residual_plot_limits = False

        plt.ylim(gdata.residual_options.min_residual,np.max(self.time_history_residual))
        
        self.line_residual_0.set_xdata(self.time_history_residual_N)
        data = self.line_residual_0.get_ydata()
        data = np.append(data,self.time_history_residual[-1][0])
        self.line_residual_0.set_ydata(data)
        
        self.line_residual_1.set_xdata(self.time_history_residual_N)
        data = self.line_residual_1.get_ydata()
        data = np.append(data,self.time_history_residual[-1][1])
        self.line_residual_1.set_ydata(data)
        
        self.line_residual_2.set_xdata(self.time_history_residual_N)
        data = self.line_residual_2.get_ydata()
        data = np.append(data,self.time_history_residual[-1][2])
        self.line_residual_2.set_ydata(data)
        
        self.line_residual_3.set_xdata(self.time_history_residual_N)
        data = self.line_residual_3.get_ydata()
        data = np.append(data,self.time_history_residual[-1][3])
        self.line_residual_3.set_ydata(data)
        
        plt.draw()
        
        print "residual = ",gdata.residual_options.global_residual

        #print "done"

        return

    def secToTime(self,seconds):
        """
        return hrs mins and secs from an input of secs
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        out = "%d:%02d:%02d" % (h, m, s)

        return out
        
    def restart(self):
        """
        open up a file that we want to restart from
        """
        
        if gdata.restart:
            # open the old hdf
            self.restart_hdf = h5py.File(gdata.restart, 'r')
            gdata.step = self.restart_hdf['global_data/final_step'][()]
            gdata.time = self.restart_hdf['step_%d/block_0/time'%gdata.step][()]
            
            if gdata.residual_options.plot_residual:
                self.time_history_residual_N = self.restart_hdf['global_data/residual_xy'][0,:].tolist()
                self.time_history_residual = self.restart_hdf['global_data/residual_xy'][1,:].tolist()
                
        else:
            self.restart_hdf = None
            
        
        return

    def initHDF(self):
        """
        initialise HDF5 file
        """
        
        print "initialising HDF5...",
        
        save = gdata.save_options
        
         # file
        (dirName,firstName) = os.path.split(gdata.rootName)
        h5Path = os.path.join(gdata.rootName,"HDF5")
        if not os.access(h5Path, os.F_OK):
            os.makedirs(h5Path)

        
        if save.save_name != "":
            name = save.save_name
        else:
            name = firstName
            
        h5Name = os.path.join(h5Path,name+".h5")
            
        if gdata.restart & os.path.isfile(h5Name):
            # we can't over-write our restart file!!
            name += '_restart_t=%g'%gdata.time
            name.replace(".","-")
        
        h5Name = os.path.join(h5Path,name+".h5") 
        save.h5Name = h5Name
        self.h5Name = h5Name
        self.h5name_short = name+".h5"
            
        
        self.hdf = h5py.File(h5Name, 'w') #open new file to save to
        
        grp = self.hdf.create_group("global_data")
        
        grp.create_dataset("run_config",data=gdata.config_string)
        
        grp.create_dataset("device",data=gdata.device)
        grp.create_dataset("platform",data=gdata.platform)
        grp.create_dataset("title",data=gdata.title)
        grp.create_dataset("CFL",data=gdata.CFL)
        grp.create_dataset("t0",data=gdata.t0)
        grp.create_dataset("Nv",data=gdata.Nv)
        grp.create_dataset("chi",data=gdata.chi)
        grp.create_dataset("R",data=gdata.R)
        grp.create_dataset("gamma",data=gdata.gamma)
        grp.create_dataset("Pr",data=gdata.Pr)
        grp.create_dataset("Kn",data=gdata.Kn)
        grp.create_dataset("L_ref",data=gdata.L_ref)
        grp.create_dataset("D_ref",data=gdata.D_ref)
        grp.create_dataset("T_ref",data=gdata.T_ref)
        grp.create_dataset("C_ref",data=gdata.C_ref)
        grp.create_dataset("t_ref",data=gdata.t_ref)
        grp.create_dataset("b",data=gdata.b)
        grp.create_dataset("K",data=gdata.K)
        grp.create_dataset("Nb",data=len(self.blocks))
        grp.create_dataset("skip",data=save.save_count)
        grp.create_dataset("quad",data=gdata.quad)
        grp.create_dataset("weight",data=gdata.weight)
        
        if self.restart_hdf:
            grp.create_dataset("restart",data=gdata.restart)
        
        # save block constant data
        self.xdmf_blocks = []
        for b in self.blocks:
            blk = grp.create_group("block_"+str(b.id))
            blk.create_dataset("Ni",data=b.Ni)
            blk.create_dataset("Nj",data=b.Nj)
            blk.create_dataset("ni",data=b.ni)
            blk.create_dataset("nj",data=b.nj)
            blk.create_dataset("ghost",data=b.ghost)
            blk.create_dataset("x",data=b.x, compression=save.compression)
            blk.create_dataset("y",data=b.y, compression=save.compression)
            blk.create_dataset("centreX",data=b.centreX, compression=save.compression)
            blk.create_dataset("centreY",data=b.centreY, compression=save.compression)
            
            s = ""
            s += '<Grid Name="Block_%d" GridType="Uniform">\n'%(b.id)
            s += '<Topology TopologyType="2DSMesh" NumberOfElements="%d %d"/>\n'%(b.ni+1, b.nj+1)
            s += '<Geometry GeometryType="X_Y">\n'
            s += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(b.ni+1, b.nj+1)
            s += '%s:/global_data/block_%d/x\n'%(self.h5name_short, b.id)
            s += '</DataItem>\n'
            s += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(b.ni+1, b.nj+1)
            s += '%s:/global_data/block_%d/y\n'%(self.h5name_short, b.id)
            s += '</DataItem>\n'
            s += '</Geometry>\n'
            self.xdmf_blocks.append(s)
            
        #####################################################################
        ## The xdmf file for reading the hdf5 file into paraview
        # or similar
        xdmf_name = os.path.splitext(self.h5Name)[0]
        self.xdmf = open(xdmf_name+'.xmf','w')
        self.xdmf.write('<?xml version="1.0" ?>\n')
        self.xdmf.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        self.xdmf.write('<Xdmf Version="2.2">)\n')
        self.xdmf.write('<Domain>\n')
        self.xdmf.write('<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')
        
        self.HDF_init = True
        
        print "done"
        
        return

    def saveHDF5(self, saveAll = False):
        """
        save data to a hdf5 file
        """
        
        if not self.HDF_init:
            self.initHDF()
        
        print "saving to HDF5...",
        
        grp = self.hdf.require_group("step_" + str(self.step))
        
        self.xdmf.write('<Grid Name="TimeSlice" GridType="Collection" CollectionType="Spatial">\n')
        self.xdmf.write('<Time Value="%0.15f" />\n'%gdata.get_time())
        
        for b in self.blocks:
            xdmf_string = b.save_hdf(self.h5name_short, grp, self.step, all_data=saveAll)
            self.xdmf.write(self.xdmf_blocks[b.id])
            self.xdmf.write(xdmf_string)
            self.xdmf.write('</Grid>\n')
            
        self.xdmf.write('</Grid>\n')
        
        print "done"
        
        return
    
    def close_HDF5(self):
        """
        close the HDF5 file opened by init_HDF5()
        """
        
        if (not self.closed) & self.HDF_init:
            
            self.hdf.create_dataset("global_data/final_step",data=self.step)
            
            if gdata.residual_options.get_residual & (len(self.time_history_residual_N) != 0):
                length = len(self.time_history_residual_N)
                residual_xy = np.zeros((length,5))
                residual_xy[:,0] = self.time_history_residual_N
                residual_xy[:,1:5] = self.time_history_residual
                self.hdf.create_dataset("global_data/residual_xy",data=residual_xy, compression=gdata.save_options.compression)
            
            self.xdmf.write('</Grid>\n')
            self.xdmf.write('</Domain>\n')
            self.xdmf.write('</Xdmf>\n')
            self.xdmf.close()
            
            print "closing HDF5 file...",
            
            self.hdf.close()
            self.closed = True
        
        print "done"
        
        return

    def saveToFile(self, close_file=False, save_f=False):
        """
        group saving calls together
        """
        if gdata.save_options.save & (not self.saved):
            
            print "saving step: ",self.step
            self.saveHDF5(save_f)
            
            self.saved = True
            
        if close_file:
            self.close_HDF5()

        return
        
    def total_mass(self):
        """
        calculate the total mass contained in this simulation
        """
        
        mass = 0.0
        
        for b in self.blocks:
            mass += b.block_mass()
            
        return mass
        
    def __del__(self):
        """
        clean up on exit
        """
        if self.HDF_init:
            self.close_HDF5()
        
        return
        