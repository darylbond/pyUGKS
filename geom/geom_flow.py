# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:29:46 2011

@author: uqdbond1
"""

#LB_flow.py

class FlowCondition(object):
    """
    Python class to organise the setting of each flow condition.
    
    D = density, kg/m^3
    P: static pressure, Pa
    U: x-component of velocity, m/s
    V: y-component of velocity, m/s
    W: z-component of velocity, m/s
    T: average temperature, K
    P: pressure, Pa (only used for definition of boundary condition)
    label: (optional) string label
    UDF_* : user defined functions -> ie D = (x + y)
    """
    
    flowList = []

    __slots__ = 'D', 'U', 'V', 'T', 'P', 'qx', 'qy', 'tau', 'label', 'indx',\
    'UDF_D','UDF_U','UDF_V', 'UDF_T', 'UDF_qx', 'UDF_qy', 'UDF_tau',\
    'isUDF', 'isConst',
    
    def __init__(self, D=1.0, U=0.0, V=0.0, T=300.0, P=-1, qx=0.0, qy=0.0, tau=0.0,
                 label="", UDF_D="", UDF_U="", UDF_V="", UDF_T="", 
                 UDF_qx="", UDF_qy="", UDF_tau=""):
        """
        Create a FlowCondition.
        """
        self.D = D
        self.U = U
        self.V = V
        self.T = T
        self.P = P
        self.qx = qx
        self.qy = qy
        self.tau = tau
        
        self.UDF_D = UDF_D
        self.UDF_U = UDF_U
        self.UDF_V = UDF_V
        self.UDF_T = UDF_T
        self.UDF_qx = UDF_qx
        self.UDF_qy = UDF_qy
        self.UDF_tau = UDF_tau
        
        if (not UDF_D) | (not UDF_U) | (not UDF_V) | (not UDF_T):
            self.isUDF = True
        else:
            self.isUDF = False
        
        self.label = label
    
        self.indx = len(FlowCondition.flowList) # next available index
        FlowCondition.flowList.append(self)

        return

    def __str__(self):
        """
        Produce a string representation that can be used by str() and print.
        """
        str = "FlowCondition("
        str += "D=%g, U=%g, V=%g, T = %g" % (self.D, self.U, self.V, self.T)
        str += ", label=\"" + self.label + "\""
        str += " UDF_D = " + self.UDF_D
        str += " UDF_U = " + self.UDF_U
        str += " UDF_V = " + self.UDF_V
        str += " UDF_T = " + self.UDF_T + ")"
        return str
