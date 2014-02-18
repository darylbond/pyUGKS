#ugks_sim.py

#system
import pyopencl as cl
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import time
import os
#from math import sqrt
import h5py

#local
from ugks_data import gdata
from ugks_data import Block
from ugks_block import UGKSBlock

from geom.geom_bc_defs import *
from geom.geom_defs import *

def reject_outliers(data, m=2):
    data = np.array(data)
    return data[abs(data - np.mean(data)) < m * np.std(data)]

class UGKSim(object):
    """
    class that handles actual simulation
    """

    step = 0
    
    run_time = []
    opt = []
    opt_time = []
    opt_step = 0
    opt_item = 1
    force_time = False
    
    saved = False
    closed = False
    HDF_init = False
    picName = False
    monitorName = False
    pic_counter = 0

    def __init__(self):
        """
        initialise class
        """
        
        self.runtime_variables()

        ## OpenCL initialisation
        print "\n INITIALISE OPENCL\n"
        self.init_CL()
        
        # initialise the optimization list to run through
        # also sets the values for compilation in CL source
        self.set_opt_list()
        
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
        
        print "\n INITIALISE GHOST CELLS\n"
        self.updateAllBC()
        
        print "\n INITIALISE TIME STEP\n"
        self.calcTimeStep()
        self.updateTimeStep(limit=False)

        print "\n INITIALISATION COMPLETE\n"
                
        
        if self.restart_hdf:
            self.restart_hdf.close()

        # flags
        self.saved = False
        self.closed = False
        self.HDF_init = False
        self.clock_time_stop_seconds = gdata.clock_time_stop[0]*60**2 + gdata.clock_time_stop[1]*60
        
        # save to file
        self.saveToFile(save_f=gdata.save_options.save_initial_f)
        
        # pics
        if not self.picName:
            save = gdata.save_options
            
            # file
            (dirName,firstName) = os.path.split(gdata.rootName)
            
            # file name
            if save.save_name != "":
                name = save.save_name
            else:
                name = firstName
                
            
            picPath = os.path.join(gdata.rootName,"IMG/"+name)
            if not os.access(picPath, os.F_OK):
                os.makedirs(picPath)
            
            self.picName = os.path.join(picPath,name)
            self.monitorName = os.path.join(gdata.rootName, "IMG", firstName)

        return
        
    def runtime_variables(self):
        """
        """
        self.res_slope_N = []
        self.res_slope = [[],[],[],[]]
        self.time_history_residual = []
        self.time_history_residual_N = []
        
    def set_opt_list(self):
        """
        set up the list of settings to run through for optimisation
        of run time
        """
        
        if not gdata.opt_run:
            return
        
        opt = []
        
        r = int(np.log2(256)+1)
        s = int(np.log2(gdata.opt_start))
        
        if gdata.CL_local_size == None:
            cl_range = range(s,r)
        else:
            cl_range = [0]
        
        for i in cl_range:
            
            if gdata.CL_local_size == None:
                cl_local_size = 2**i
            else:
                cl_local_size = gdata.CL_local_size
            
            if cl_local_size <= gdata.Nv:
                for j in range(s,r):
                    for k in range(s,r):
                        si = 2**j
                        sj = 2**k
                        if (si*sj) <= 256:
                            sub_opt = [cl_local_size, si, sj]
                            opt.append(sub_opt)
                            
        self.opt = opt
        
        gdata.CL_local_size = opt[0][0]
        gdata.work_size_i = opt[0][1]
        gdata.work_size_j = opt[0][2]
        
        return
        
    def start_timer(self):
        """
        start timer
        """
        
        self.timer = time.time()
        
        return
        
    def stop_timer(self):
        """
        return timer
        """
        
        return time.time() - self.timer

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
        
    def calcTimeStep(self):
        """
        calculate the maximum allowable time step for each block based on 
        hydro-dynamic properties
        """

        for b in self.blocks:
            b.getDT()

        return
        
    def updateTimeStep(self, limit=True):
        """
        update the global time step
        """
        
        
        
        if not self.force_time:
        
            dt_old = gdata.dt
            
            dt_list = []
    
            for b in self.blocks:
                dt_list.append(b.max_dt)
                
            dt_new = min(dt_list)
            
            if dt_old != dt_new:
                print "dt = %g -->"%gdata.dt,
                # impose a limit on how much the time step can increase for each iteration
                if (dt_new > (1.0+gdata.delta_dt)*dt_old) & limit:
                    dt_new = (1.0+gdata.delta_dt)*dt_old
                gdata.dt = dt_new
                print "dt = %g"%gdata.dt

        return

    def updateAllBC(self):
        """
        update the ghost cells of all blocks
        """

        for b in self.blocks:
            b.updateBC()

        return
    
    def update_run_config(self, i):
        """
        update the runtime configuration
        """
        
        gdata.work_size_i = self.opt[i][1]
        gdata.work_size_j = self.opt[i][2]
        
        if gdata.CL_local_size != self.opt[i][0]:
            print "opt -> updating source"
            gdata.CL_local_size = self.opt[i][0]
            for b in self.blocks:
                b.update_CL()
        
        return          
        
    def run_optimisation(self):
        """
        run optimisation
        """
        
        if gdata.opt_run:
            self.queue.finish()
            self.run_time.append(self.stop_timer())
            self.opt_step += 1
            if self.opt_step == gdata.opt_sample_size:
                self.run_time = reject_outliers(self.run_time)
                self.opt_time.append(np.sum(self.run_time)/float(self.opt_step))
                
                print "opt -> [%d, %d, %d] = %s"%(gdata.CL_local_size,
                                            gdata.work_size_i,
                                            gdata.work_size_j,
                                            self.secToTime(self.opt_time[-1]))

                if self.opt_item == len(self.opt):
                    gdata.opt_run == False
                    i = self.opt_time.index(min(self.opt_time))

                    print "--optimum found--"                    
                    
                    self.update_run_config(i)
                    
                    print "optimum -> [%d, %d, %d] = %s"%(gdata.CL_local_size,
                                            gdata.work_size_i,
                                            gdata.work_size_j,
                                            self.secToTime(self.opt_time[i]))
                                            
                    gdata.opt_run = False
                                            
#                    fig = plt.figure()
#                    ax = fig.add_subplot(111, projection='3d')
#                    data = np.array(self.opt)
#                    sc = ax.scatter(data[:,1],data[:,2],self.opt_time,c=data[:,0])
#                    plt.colorbar(sc)
#                    ax.set_xlabel('i')
#                    ax.set_ylabel('j')
#                    ax.set_zlabel('t')
#                    plt.show()
                    
                    
                else:
                
                    self.update_run_config(self.opt_item)
                                       
                    self.run_time = []
                    self.opt_step = 0
                    self.opt_item += 1
                
                
        return
        
    def one_step(self, get_dt, get_res):
        """
        run one step of the simulation
        """
        
        if get_res:
            res = 0.0
        
        if gdata.opt_run:
            self.start_timer()
        
        cl.enqueue_barrier(self.queue)
        
        # update the parametric boundary conditions that have a time component
        for b in self.blocks:
            b.parametricBC()

        # check if the required time step has changed
        # this may have occured due to the adsorbing wall needing a shorter
        # time step to avoid overflow
        dt_old = 0.0
        count = 0
        while (gdata.dt != dt_old) & (count < 2):
            dt_old = gdata.dt
            for b in self.blocks:
                b.UGKS_flux()
            cl.enqueue_barrier(self.queue)
            self.updateTimeStep()
            count += 1
            
        if count > 1:
            if gdata.step == 1:
                raise RuntimeError("Overshot on first iteration, decrease time step")
            get_dt = True

        
        for b in self.blocks:
            b.UGKS_update(get_res)
            
        cl.enqueue_barrier(self.queue)
        
        self.run_optimisation()
        
        if get_dt:
            self.calcTimeStep()
            self.updateTimeStep()

        if get_res:
            for b in self.blocks:
                block_res = b.residual
                res += block_res
            res /= len(self.blocks)

        # update time counter - lattice units
        gdata.time += gdata.dt

        if get_res:
            gdata.residual_options.global_residual = res
            gdata.residual_options.residual_history.append(res)
        
        
        self.step += 1
        gdata.step = self.step
        
        self.queue.flush()
        
        # make sure all ghost cells are up to date
        self.updateAllBC()

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
        if res.get_residual:
            below_slope_threshold_count = 0
            first_res = True
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
            self.saved = False
        
        while self.step < gdata.max_step:
            
            ## FLAGS
            if res.get_residual:
                if (res.non_linear_output & (self.step%mag == 0)) & (self.step < res.non_linear_output_limit):
                    mag = int(10**(np.floor(np.log10(self.step+1))))
                    res.residual_count = mag
                    if res.non_linear_save:
                        save.save_count = mag
                    if res.non_linear_dt:
                        gdata.dt_update_count = mag
                        
                        
                if gdata.restart:
                    if self.step >= res.non_linear_output_limit:
                        if res.non_linear_save:
                            save.save_count = res.non_linear_output_limit
                        if res.non_linear_dt:
                            gdata.dt_update_count = res.non_linear_output_limit
            
            if res.get_residual & ((self.step+1)%res.residual_count == 0):
                get_res = True
            else:
                get_res = False
            
            if gdata.dt_update_count < 0:
                if self.step == -gdata.dt_update_count:
                    get_dt = True
                else:
                    get_dt = False
            elif not (self.step+1) % gdata.dt_update_count:
                get_dt = True
            else:
                get_dt = False
                
            # one iteration of the method
            step_finished = self.one_step(get_dt, get_res)
            
            self.saved = False

            gdata.read_conf()
                    
            ###
            # CONDITIONALS
            ###
            if (gdata.run_stop_script_count != -1):
                if not self.step % gdata.run_stop_script_count:
                    print "step ",self.step
                    exec gdata.stop_script
            
            if gdata.exit:
                print "simulation exit -> interrupted"
                print "step ",self.step," t = ",gdata.time
                break
            
            # exit condition: sim has generated a NaN
            if foundNaN:
                print "simulation exit -> NaN encountered"
                print "step ",self.step," t = ",gdata.time
                break
            
            # exit condition maximum time reached
            if gdata.time >= gdata.max_time:
                print "simulation exit -> time limit reached"
                print "step ",self.step," t = ",gdata.time
                break
            
            if (time.time()-t0) >= self.clock_time_stop_seconds:
                print "simulation exit -> clock time limit reached"
                print "step ",self.step," t = ",gdata.time
                break
            
            
            if get_res:
                self.time_history_residual.append(res.global_residual)
                self.time_history_residual_N.append(self.step)
                if (len(self.time_history_residual) >= res.slope_sample) & (self.step >= res.slope_start):
                    slope = self.get_residual_slope()
                    print "residual slope = ",slope
                    if np.all(slope < res.min_slope):
                        below_slope_threshold_count += 1
                    if below_slope_threshold_count > 2:      
                        print "simulation exit -> residual stabilised"
                        print "step ",self.step," t = ",gdata.time
                        break
                if res.plot_residual:
                    self.plotResidualUpdate()
                if self.step >= res.residual_start:
                    if first_res:
                        initial_res = res.global_residual
                        first_res = False
                    else:
                        res_drop = initial_res/res.global_residual
                        print "residual drop = ",res_drop
                        if np.all(res_drop >= res.res_drop):
                            print "simulation exit -> residual reduction limit reached"
                            print "step ",self.step," t = ",gdata.time
                            break
                    if np.all(res.global_residual <= res.min_residual):
                        print "simulation exit -> minimum residual reached"
                        print "step ",self.step," t = ",gdata.time
                        break
                
            
            # print progress
            if ((not self.step % gdata.print_count) & (gdata.print_count > 0)) | get_res:
                step_finished.wait()
                print "step %d, t = %0.5f"%(self.step, gdata.time)
                if not res.get_residual:
                    sec_per_step = (time.time()-t0)/float(self.step)
                    steps_remaining_time = (gdata.max_time - gdata.time)/gdata.dt
                    steps_remaining_steps = gdata.max_step - self.step
                    steps_remaining = min(steps_remaining_time,steps_remaining_steps)
                    time_remaining = steps_remaining*sec_per_step
                
                    print " --> t_final-t_now = %s"%self.secToTime(time_remaining)
            
            # save to file
            if (not self.step % save.save_count) & (self.step <= gdata.max_step-1):
                step_finished.wait()
                print "saving"
                if self.step == gdata.max_step:
                    self.saveToFile(save_f=save.save_final_f)
                else:
                    self.saveToFile(save_f=save.save_f_always)
                    
                if save.save_pic:
                    self.plot_step()
                    
            if gdata.time < save.initial_save_cutoff_time:
                if not self.step % save.initial_save_count:
                    step_finished.wait()
                    self.saveToFile()
                
            
            if (gdata.max_time - gdata.time) < gdata.dt:
                gdata.dt = gdata.max_time - gdata.time
                self.force_time = True
        
        self.queue.finish()        
        
        ##
        # MAIN LOOP END
        ##
        
        if self.step >= gdata.max_step:
            print "simulation exit -> step count reached"
            print " -> step ",self.step," t = ",gdata.time
        
        print "wall clock time of simulation run = ",self.secToTime(time.time()-t0)
        
        self.saveToFile(close_file=True, save_f=save.save_final_f, save_wall_flux=save.save_final_flux)

        if res.plot_residual & res.get_residual:
            plt.ioff()
            if res.hold_plot:
                print "close residual plot to end run"
                plt.show()
            else:
                plt.close()
                
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
        self.resPlot = self.resFig.add_subplot(111)
        
        self.line_slope_0, = self.resPlot.loglog(self.res_slope_N, self.res_slope[0],'r--',label="rho")
        self.line_slope_1, = self.resPlot.loglog(self.res_slope_N, self.res_slope[1], 'g--', label="U")
        self.line_slope_2, = self.resPlot.loglog(self.res_slope_N, self.res_slope[2], 'b--', label="V")
        self.line_slope_3, = self.resPlot.loglog(self.res_slope_N, self.res_slope[3], 'k--', label="1/T")
        
        if gdata.restart:
            init = np.array(self.time_history_residual)
            self.line_residual_0, = self.resPlot.loglog(self.time_history_residual_N, init[:,0],'r',label="rho")
            self.line_residual_1, = self.resPlot.loglog(self.time_history_residual_N, init[:,1], 'g', label="U")
            self.line_residual_2, = self.resPlot.loglog(self.time_history_residual_N, init[:,2], 'b', label="V")
            self.line_residual_3, = self.resPlot.loglog(self.time_history_residual_N, init[:,3], 'k', label="1/T")
        else:
            self.line_residual_0, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual,'r',label="rho")
            self.line_residual_1, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual, 'g', label="U")
            self.line_residual_2, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual, 'b', label="V")
            self.line_residual_3, = self.resPlot.loglog(self.time_history_residual_N, self.time_history_residual, 'k', label="1/T")
        
        self.resPlot.legend(loc=3,ncol=2)        
        
        if not self.time_history_residual:
            max_res = 1
        else:
            max_res = np.max(self.time_history_residual)
        
        plot_title = "Residual"
        if gdata.title:
            plot_title += " - " + gdata.title
        self.resPlot.set_title(plot_title)
        
        self.resPlot.grid(True)
        self.residual_plot_limits = True
        self.resPlot.set_xlim(1,gdata.max_step)
        self.resPlot.set_ylim(gdata.residual_options.min_residual,max_res)
        
        name = self.picName+"_RESIDUAL.png"
        self.resFig.savefig(name)
        
        name = self.monitorName+"_RESIDUAL.png"
        self.resFig.savefig(name)
            
        
        print "\nresidual plot generated\n"
        

        # throw away first few samples
        self.res_throw = True

        return

    def plotResidualUpdate(self):
        """
        update the residual plot
        """
        
        if not self.resPlot:
            return
            
        max_y = np.max(self.time_history_residual)

        self.resPlot.set_ylim(gdata.residual_options.min_residual,max_y)
        self.resPlot.set_xlim(np.min(self.time_history_residual_N),gdata.max_step)
        
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
        
        ###
        if len(self.res_slope_N) > 1:
            
            max_y = max(max_y, np.max(self.res_slope))

            self.resPlot.set_ylim(gdata.residual_options.min_residual,max_y)
            
            self.line_slope_0.set_xdata(self.res_slope_N)
            self.line_slope_0.set_ydata(self.res_slope[0])
            
            self.line_slope_1.set_xdata(self.res_slope_N)
            self.line_slope_1.set_ydata(self.res_slope[1])
            
            self.line_slope_2.set_xdata(self.res_slope_N)
            self.line_slope_2.set_ydata(self.res_slope[2])
            
            self.line_slope_3.set_xdata(self.res_slope_N)
            self.line_slope_3.set_ydata(self.res_slope[3])
            
        
        plt.draw()
        
        name = self.picName+"_RESIDUAL.png"
        self.resFig.savefig(name)
        
        name = self.monitorName+"_RESIDUAL.png"
        self.resFig.savefig(name)
        
        print "residual = ",gdata.residual_options.global_residual

        #print "done"

        return
    
    def get_residual_slope(self):
        """
        calculate the slope of the residual
        """
        # first normalize
        
        N = gdata.residual_options.slope_sample
        
        if len(self.time_history_residual_N) >= N:
            
            res = np.array(self.time_history_residual)
            
            # check if we have zeros, or negatives, in our residuals
            if np.any(res <= 0.0):
                return np.inf
            
            res = np.log10(res)
            
            for i in range(4):
                maxi = np.max(res[:,i])
                mini = np.min(res[:,i])
                res[:,i] = (res[:,i] - mini)/(maxi - mini) + 1.0
        
            res = res[-N::,:]
            res_x = np.log10(self.time_history_residual_N[-N::])

            slope = np.polyfit(res_x, res, 1)
            slope = np.abs(slope[0])
            
            self.res_slope_N.append(self.step)
            for i in range(4):
                self.res_slope[i].append(slope[i])
                
            return slope
            
        return np.inf

    def secToTime(self,seconds):
        """
        return hrs mins and secs from an input of secs
        """
        s, ms = divmod(seconds*1000, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        out = "%d:%02d:%02d:%02d" % (h, m, s, ms)

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
            
            if gdata.residual_options.get_residual:
                self.time_history_residual_N = self.restart_hdf['global_data/residual_xy'][:,0].tolist()
                self.time_history_residual = self.restart_hdf['global_data/residual_xy'][:,1:5].tolist()
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
            
        if (gdata.restart != False) & os.path.isfile(h5Name):
            # we can't over-write our restart file!!
            name += '_restart_t=%g'%gdata.time
            name.replace(".","-")
        
        h5Name = os.path.join(h5Path,name+".h5") 
        save.h5Name = h5Name
        self.h5Name = h5Name
        self.h5name_short = name+".h5"
            
        
        print "HDF save to --> %s"%h5Name
        self.hdf = h5py.File(h5Name, 'w') #open new file to save to
        
        grp = self.hdf.create_group("global_data")
        
        grp.create_dataset("run_config",data=gdata.config_string)
        
        grp.create_dataset("device",data=gdata.device)
        grp.create_dataset("platform",data=gdata.platform)
        grp.create_dataset("title",data=gdata.title)
        grp.create_dataset("CFL",data=gdata.CFL)
        grp.create_dataset("t0",data=gdata.t0)
        grp.create_dataset("Nv",data=gdata.Nv)
        grp.create_dataset("mu_ref",data=gdata.mu_ref)
        grp.create_dataset("Kn_eff",data=gdata.Kn_eff)
        grp.create_dataset("omega",data=gdata.omega)
        grp.create_dataset("R",data=gdata.R)
        grp.create_dataset("gamma",data=gdata.gamma)
        grp.create_dataset("Pr",data=gdata.Pr)
        grp.create_dataset("L_ref",data=gdata.L_ref)
        grp.create_dataset("D_ref",data=gdata.D_ref)
        grp.create_dataset("T_ref",data=gdata.T_ref)
        grp.create_dataset("C_ref",data=gdata.C_ref)
        grp.create_dataset("t_ref",data=gdata.t_ref)
        grp.create_dataset("b",data=gdata.b)
        grp.create_dataset("K",data=gdata.K)
        grp.create_dataset("Nb",data=len(self.blocks))
        grp.create_dataset("quad",data=gdata.quad)
        grp.create_dataset("weight",data=gdata.weight)
        
        if self.restart_hdf:
            grp.create_dataset("restart",data=gdata.restart)
        
        # save block constant data
        self.xdmf_blocks = []
        for b in self.blocks:
            blk = grp.create_group("block_"+str(b.id))
            blk.create_dataset("label",data=b.label)
            blk.create_dataset("Ni",data=b.Ni)
            blk.create_dataset("Nj",data=b.Nj)
            blk.create_dataset("ni",data=b.ni)
            blk.create_dataset("nj",data=b.nj)
            blk.create_dataset("ghost",data=b.ghost)
            blk.create_dataset("x",data=b.x, compression=save.compression)
            blk.create_dataset("y",data=b.y, compression=save.compression)
            blk.create_dataset("xy",data=b.xy_H, compression=save.compression)
            blk.create_dataset("centreX",data=b.centreX, compression=save.compression)
            blk.create_dataset("centreY",data=b.centreY, compression=save.compression)
            
            bcg = blk.create_group('BC')
            nc = [b.ni, b.nj, b.ni, b.nj]
            for face, bc in enumerate(b.bc_list):
                bcf = bcg.create_group(faceName[face])
                bcf.create_dataset('type_of_BC',data=bcName[bc.type_of_BC])
                bcf.create_dataset('other_block',data=bc.other_block)
                bcf.create_dataset('other_face',data=bc.other_face)
                bcf.create_dataset('orientation',data=bc.orientation)
                bcf.create_dataset('label',data=bc.label)
                bcf.create_dataset('D',data=bc.D)
                bcf.create_dataset('U',data=bc.U)
                bcf.create_dataset('V',data=bc.V)
                bcf.create_dataset('T',data=bc.T)
                bcf.create_dataset('P',data=bc.P)
                if bc.UDF_D:
                    bcf.create_dataset('UDF_D',data=bc.UDF_D)
                if bc.UDF_U:
                    bcf.create_dataset('UDF_U',data=bc.UDF_U)
                if bc.UDF_V:
                    bcf.create_dataset('UDF_V',data=bc.UDF_V)
                if bc.UDF_T:
                    bcf.create_dataset('UDF_T',data=bc.UDF_T)
                
                if bc.type_of_BC == ADSORBING:
                    bcf.create_dataset('adsorb',data=bc.adsorb)
                    bcf.create_dataset('beta_n',data=bc.beta_n)
                    bcf.create_dataset('beta_t',data=bc.beta_t)
                    bcf.create_dataset('alpha_n',data=bc.alpha_t)
                    bcf.create_dataset('alpha_t',data=bc.alpha_t)
                    bcf.create_dataset('k_f',data=bc.k_f)
                    bcf.create_dataset('S_T',data=bc.S_T)
                    bcf.create_dataset('alpha_p',data=bc.alpha_p)
                    bcf.create_dataset('cover_initial',data=bc.cover_initial)
                    bcf.create_dataset('reflect_type',data=bc.reflect_type)
                
            
                # edge coords
                sz = nc[face]+1
                ex = np.zeros((sz), dtype=np.float64)
                ey = np.zeros((sz), dtype=np.float64)
                
                if face == 0:
                    ex[:] = b.x[:,-1,0]
                    ey[:] = b.y[:,-1,0]
                elif face == 1:
                    ex[:] = b.x[-1,:,0]
                    ey[:] = b.y[-1,:,0]
                elif face == 2:
                    ex[:] = b.x[:,0,0]
                    ey[:] = b.y[:,0,0]
                elif face == 3:
                    ex[:] = b.x[0,:,0]
                    ey[:] = b.y[0,:,0]
                
                bcf.create_dataset("x_edge",data=ex, compression=save.compression)
                bcf.create_dataset("y_edge",data=ey, compression=save.compression)
            
            
            
            s1 = ""
            s1 += '<Grid Name="Block_%d" GridType="Uniform">\n'%(b.id)
            s1 += '<Topology TopologyType="2DSMesh" NumberOfElements="%d %d"/>\n'%(b.ni+1, b.nj+1)
            s1 += '<Geometry GeometryType="X_Y">\n'
            s1 += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(b.ni+1, b.nj+1)
            s1 += '%s:/global_data/block_%d/x\n'%(self.h5name_short, b.id)
            s1 += '</DataItem>\n'
            s1 += '<DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">\n'%(b.ni+1, b.nj+1)
            s1 += '%s:/global_data/block_%d/y\n'%(self.h5name_short, b.id)
            s1 += '</DataItem>\n'
            s1 += '</Geometry>\n'
            
            self.xdmf_blocks.append(s1)
            
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

    def saveHDF5(self, save_f=False, save_flux=False):
        """
        save data to a hdf5 file
        """
        
        if not self.HDF_init:
            self.initHDF()
        
        print "saving to HDF5...",
        
        grp = self.hdf.require_group("step_" + str(self.step))
        grp.create_dataset('step',data=self.step)
        
        self.xdmf.write('<Grid Name="TimeSlice" GridType="Collection" CollectionType="Spatial">\n')
        self.xdmf.write('<Time Value="%0.15f" />\n'%gdata.time)
        
        for b in self.blocks:
            xdmf_string = b.save_hdf(self.h5name_short, grp, self.step, save_f=save_f, save_flux=save_flux)
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

    def saveToFile(self, close_file=False, save_f=False, save_wall_flux=False):
        """
        group saving calls together
        """
        if gdata.save_options.save & (not self.saved):
            
            print "saving step: ",self.step
            self.saveHDF5(save_f, save_wall_flux)
            
            self.saved = True
            
        if close_file:
            self.close_HDF5()

        return
        
    def total_mass(self):
        """
        calculate the total mass contained in this simulation
        """
        bulk_mass = 0.0
        adsorbed_mass = 0.0
        
        for b in self.blocks:
            bulk_mass_i, adsorbed_mass_i = b.block_mass()
            bulk_mass += bulk_mass_i
            adsorbed_mass += adsorbed_mass_i
            
        return bulk_mass, adsorbed_mass
        
    def plot_step(self):
        """
        save a plot of the current time step to png
        """
        
        X = []
        Y = []
        Z = []
        
        max_z = -1
        min_z = 1
        
        save = gdata.save_options
        
        for block in self.blocks:
            x = block.x[:,:,0]
            y = block.y[:,:,0]
            
            block.updateHost()
            
            if save.pic_type == "UV":
                UV = block.macro_H[:,:,1:3]
                z = np.sqrt(UV[:,:,0]**2 + UV[:,:,1]**2)
            elif save.pic_type == "D":
                z = block.macro_H[:,:,0]
            elif save.pic_type == "T":
                z = 1.0/block.macro_H[:,:,3]
            elif save.pic_type == "P":
                z = 0.5*block.macro_H[:,:,0]*block.macro_H[:,:,3]
            elif save.pic_type == "Q":
                Q = block.Q_H
                z = np.sqrt(Q[:,:,0]**2 + Q[:,:,1]**2)
                
            X.append(x)
            Y.append(y)
            Z.append(z)
            
            max_z = max(max_z, np.max(z))
            min_z = min(min_z, np.min(z))
            
        max_size = 12
        
        fig = plt.figure(figsize=(max_size, max_size), dpi=300)
        ax = fig.add_subplot(111)
        
        for i in range(len(X)):
            
            ax.pcolormesh(X[i], Y[i], Z[i], vmin=min_z, vmax=max_z)
            
        ax.set_aspect('equal')
            
            
        name = self.picName+"_%s_step_%0.5i.png"%(save.pic_type, self.pic_counter)
        fig.savefig(name)
        
        name = self.monitorName+"_%s_LATEST.png"%save.pic_type
        fig.savefig(name)
        
        plt.close(fig)
        
        self.pic_counter += 1
        
        return 
        
    def __del__(self):
        """
        clean up on exit
        """
        if self.HDF_init:
            self.close_HDF5()
        
        return
        