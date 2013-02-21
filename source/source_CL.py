# source_CL.py

"""
file loader for source files located in this directory
"""

import re, os

source_CL_path = os.path.dirname(os.path.realpath(__file__))

class SourceLoader(object):
    """
    container for links to source files
    """
    
    # list of source files

    source = ['clFunctions.c', 'clKernel.c', 
              'clGhostRoutines.c', 'clRK.c']
    
    def __init__(self,source_path = source_CL_path):
        """
        provide handles to all source files
        need to pass in location of source files
        """
        
        self.src = dict()
        
        defines = []
        
        print 'scanning source code...',

        for s in self.source:
            f = open(source_path+"/"+s,'r')
            f_lines = f.readlines()
            # search for any # define statements
            flag = False
            these_defines = []
            for line in f_lines:
                if re.search("#define",line):
                    flag = True
                    these_defines.append(line)
            
            if flag:
                for line in these_defines:
                    f_lines.remove(line)
            
            defines += these_defines
            
            fstr = "".join(f_lines)
            
            self.src[s] = fstr
            
        self.src['definitions'] = "".join(defines)
        
        print 'done'
        
        return
