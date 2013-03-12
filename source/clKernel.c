/////////////////////////////////////////
//MACROS 
/////////////////////////////////////////\n
#define F(i,j,v) Fin[NV*NJ*(i) + NV*(j) + (v)] 
#define TXYZ(i,j) Txyz[(i)*nj + (j)] 
#define XY(i,j) xy[(i)*NY + (j)] 


#define MACRO(i,j) macro[(i)*nj + (j)]
#define GD(i,j) macro[(i)*nj + (j)].s0
#define GUV(i,j) macro[(i)*nj + (j)].s12
#define GL(i,j) macro[(i)*nj + (j)].s3
#define GQ(i,j) gQ[(i)*nj + (j)] 


#define AREA(i,j) area[(i)*NJ + (j)] 
#define CENTRE(i,j) centre[(i)*NJ + (j)] 
#define MIDSIDE(i,j,face) mid_side[(i)*NJ*2 + (j)*2 + face] 
#define NORMAL(i,j,face) normal[(i)*NJ*2 + (j)*2 + face] 
#define LENGTH(i,j,face) side_length[(i)*NJ*2 + (j)*2 + face] 

#define TSTEP(i,j) time_step[(i)*nj + (j)]

/////////////////////////////////////////
// KERNEL: cellGeom
/////////////////////////////////////////
__kernel void
cellGeom(__global double2* xy,
	 __global double* area,
	 __global double2* centre,
	 __global double2* mid_side,
	 __global double2* normal,
	 __global double* side_length)
{
  // calculate all cell information
  
  // global index
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);

  // Cell layout
  // C-----B     3-----2
  // |     |     |     |
  // |  c  |     |  c  |
  // |     |     |     |
  // D-----A     0-----1
  
  double xA = XY(i+1,j).x;
  double yA = XY(i+1,j).y;
  double xB = XY(i+1,j+1).x;
  double yB = XY(i+1,j+1).y;
  double xC = XY(i,j+1).x;
  double yC = XY(i,j+1).y;
  double xD = XY(i,j).x;
  double yD = XY(i,j).y;
  
  ////////////////
  // side normals
  double2 tS;
  double2 tW;
  
  tS.x = xA - xD;
  tS.y = yA - yD;
  tW.x = xD - xC;
  tW.y = yD - yC;
  
  double2 nS;
  double2 nW;
  
  nS.x = -tS.y;
  nS.y = tS.x;
  nW.x = -tW.y;
  nW.y = tW.x;
  
  NORMAL(i,j,SOUTH) = normalize(nS);
  NORMAL(i,j,WEST) = normalize(nW);
  
  ////////////////
  //Cell area in the (x,y)-plane.
  double xyarea = 0.5 * ((xB + xA) * (yB - yA) + (xC + xB) * (yC - yB) + 
		      (xD + xC) * (yD - yC) + (xA + xD) * (yA - yD));
  AREA(i,j) = xyarea;
  
  ////////////////
  // cell centroid
  double cx = 1.0 / (xyarea * 6.0) * 
		((yB - yA) * (xA * xA + xA * xB + xB * xB) + 
		  (yC - yB) * (xB * xB + xB * xC + xC * xC) + 
		  (yD - yC) * (xC * xC + xC * xD + xD * xD) + 
		  (yA - yD) * (xD * xD + xD * xA + xA * xA));
  if (fabs(cx) > 1.0e6) {
    cx = NAN;
  }
  CENTRE(i,j).x = cx;
  
  double cy = -1.0 / (xyarea * 6.0) * \
		((xB - xA) * (yA * yA + yA * yB + yB * yB) + 
		(xC - xB) * (yB * yB + yB * yC + yC * yC) + 
		(xD - xC) * (yC * yC + yC * yD + yD * yD) + 
		(xA - xD) * (yD * yD + yD * yA + yA * yA));
  if (fabs(cy) > 1.0e6) {
    cy = NAN;
  }
  CENTRE(i,j).y = cy;
 
  ////////////////
  //mid side position
  double xS, yS, xW, yW;
  
  xS = 0.5 * (xD + xA);
  yS = 0.5 * (yD + yA);
  xW = 0.5 * (xD + xC);
  yW = 0.5 * (yD + yC);
  
  MIDSIDE(i,j,SOUTH).x = xS;
  MIDSIDE(i,j,SOUTH).y = yS;
  MIDSIDE(i,j,WEST).x = xW;
  MIDSIDE(i,j,WEST).y = yW;
  
  ////////////////
  // side length
  double lS, lW;
  
  lS = sqrt((xD - xA)*(xD - xA) + (yD - yA)*(yD - yA));
  lW = sqrt((xC - xD)*(xC - xD) + (yC - yD)*(yC - yD));
  
  LENGTH(i,j,SOUTH) = lS;
  LENGTH(i,j,WEST) = lW;
  
  return;
}

/////////////////////////////////////////
// clFindDT
/////////////////////////////////////////
__kernel void
clFindDT(__global double2* xy, __global double* area,
         __global double4* macro, __global double* time_step) 
{
  // calculate the time step parameter for this cell
  
  size_t mi = get_global_id(0);
  size_t mj = get_global_id(1);
  
  double2 a = XY(mi+1+GHOST,mj+GHOST);
  double2 b = XY(mi+1+GHOST,mj+1+GHOST);
  double2 c = XY(mi+GHOST,mj+1+GHOST);
  double2 d = XY(mi+GHOST,mj+GHOST);
  
  double dx = fabs(a.x-b.x);
  double dx_temp = fabs(c.x-b.x);
  if (dx_temp > dx) {
    dx = dx_temp;
  }
  dx_temp = fabs(c.x-d.x);
  if (dx_temp > dx) {
    dx = dx_temp;
  }
  dx_temp = fabs(a.x-d.x);
  if (dx_temp > dx) {
    dx = dx_temp;
  }
  dx_temp = fabs(c.x-a.x);
  if (dx_temp > dx) {
    dx = dx_temp;
  }
  dx_temp = fabs(b.x-d.x);
  if (dx_temp > dx) {
    dx = dx_temp;
  }
  
  double dy = fabs(a.y-b.y);
  double dy_temp = fabs(c.y-b.y);
  if (dy_temp > dy) {
    dy = dy_temp;
  }
  dy_temp = fabs(c.y-d.y);
  if (dy_temp > dy) {
    dy = dy_temp;
  }
  dy_temp = fabs(a.y-d.y);
  if (dy_temp > dy) {
    dy = dy_temp;
  }
  dy_temp = fabs(c.y-a.y);
  if (dy_temp > dy) {
    dy = dy_temp;
  }
  dy_temp = fabs(b.y-d.y);
  if (dy_temp > dy) {
    dy = dy_temp;
  }
  
  double2 UV = GUV(mi,mj);
  
  double sos = soundSpeed(GL(mi,mj));
  
  double u = max(umax, fabs(UV.x)) + sos;
  double v = max(vmax, fabs(UV.y)) + sos;
  
  TSTEP(mi,mj) = (dx*u + dy*v)/AREA(mi+GHOST,mj+GHOST);
    
  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: getT
////////////////////////////////////////////////////////////////////////////////

__kernel void
getInternalTemp(__global double2* Fin, __global double4* Txyz)
{
  // get the three temperatures Tx, Ty and Tz, plus the combined

  // distribution functions global index
  int mi = get_global_id(0);
  int mj = get_global_id(1);
  
  if ((mi < ni) && (mj < nj)) {
  
    int gi = mi + GHOST;
    int gj = mj + GHOST;

    double2 f;
    double2 uv;
    
    double D = 0.0;
    double M1 = 0.0;  
    double M2 = 0.0;  
    double M4 = 0.0; 
    double M5 = 0.0;
    double H = 0.0;
    
    for (int i=0; i < NV; i++) {
      // velocities
      uv = QUAD[i];
      
      // distribution
      f = F(gi,gj,i);
      
      // moments
      D += f.x;
      M1 += uv.x*f.x;
      M2 += uv.y*f.x;
      M4 += uv.x*uv.x*f.x;
      M5 += uv.y*uv.y*f.x;
      H += f.y;
    }
      
    double U, V, Tx, Ty, Tz, T;
    
    U = M1/D;	// mean velocity x
    V = M2/D;	// mean velocity y
    Tx = 2*(M4/D - U*U);
    Ty = 2*(M5/D - V*V);
    Tz = 2*(H/D);
    T = (Tx + Ty + Tz)/B;
    
    TXYZ(mi,mj) = (double4)(Tx, Ty, Tz, T);
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: initFunctions
////////////////////////////////////////////////////////////////////////////////

// KERNEL FUNCTION
__kernel void
initFunctions(__global double2* Fin, 
__global double4* macro, __global double2* gQ) 
{
  // initialise functions, but only those that are within the simulation domain, does not include ghost cells
  
  // distribution functions global index
  size_t mi = get_global_id(0);
  size_t mj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  if ((mi < ni) && (mj < nj)) {

    // macroscopic properties global index
      
      size_t gi = mi + GHOST;
      size_t gj = mj + GHOST;

    // compute equilibrium functions
    
    for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
      size_t gv = li*LOCAL_SIZE+ti;
      if (gv < NV) {
        double2 uv = QUAD[gv];
        F(gi,gj,gv) = fEQ(MACRO(mi,mj), GQ(mi,mj), uv, gv);
      }
    }
  }
  
  return;
}

__kernel void
calcMacro(__global double2* Fin, 
            __global double4* macro) 
{
  // calculate the macroscopic properties

    size_t mi = get_global_id(0);
    size_t mj = get_global_id(1);
    size_t thread_id = get_local_id(2);
    
    __local double4 P[LOCAL_SIZE];
  
    if ((mi < ni) && (mj < nj)) {
  
        int gi = mi + GHOST;
        int gj = mj + GHOST;
        
        // initialise
        P[thread_id] = 0.0;
        
        double2 f, uv;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                f = F(gi,gj,gv);
                uv = QUAD[gv];
                P[thread_id].s0 += f.x;
                P[thread_id].s12 += uv*f.x;
                P[thread_id].s3 += 0.5*(dot(uv,uv)*f.x + f.y);
            }
        }
      
        // synchronise
        barrier(CLK_LOCAL_MEM_FENCE);

        // have populated the local array
        // now we sum up the elements, migrating the sum to the top of the stack
        int step = 1;
        int grab_id = 2;
        while (step < LOCAL_SIZE) {
            if (!(thread_id%grab_id)) { // assume LOCAL_SIZE is a power of two
                // reduce
                P[thread_id] += P[thread_id+step];
            }
            step *= 2;
            grab_id *= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // the sum of all the moments should be at the top of the stack

        if (thread_id == 0) {
            // convert to standard variables
            double4 prim = P[0];
            // convert to standard variables
            prim.s3 = 0.5*prim.s0/(gam-1.0)/(prim.s3 - 0.5*dot(prim.s12,prim.s12)/prim.s0);
            prim.s12 /= prim.s0;
            
            MACRO(mi, mj) = prim;
        }
    }
    
    return;
}

__kernel void
calcQ(__global double2* Fin, 
            __global double4* macro, 
            __global double2* gQ) 
{
  // calculate the macroscopic properties
  // NOTE: the heat flux must be multiplied by a factor of (4+K)/5 to 
  // convert to a 'true' value for use outside this simulation. The 
  // form used here is to match the analytical solution to the Couette
  // flow problem.

    size_t mi = get_global_id(0);
    size_t mj = get_global_id(1);
    size_t thread_id = get_local_id(2);
    
    __local double2 Q[LOCAL_SIZE];
  
    if ((mi < ni) && (mj < nj)) {
  
        int gi = mi + GHOST;
        int gj = mj + GHOST;
        
        // initialise
        Q[thread_id] = 0.0;
        
        double4 prim = MACRO(mi,mj);
        double2 f, uv;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                f = F(gi,gj,gv);
                uv = QUAD[gv];
                Q[thread_id].x += 0.5*((uv.x-prim.s1)*(dot(uv-prim.s12, uv-prim.s12)*f.x + (f.y/K)));
                Q[thread_id].y += 0.5*((uv.y-prim.s2)*(dot(uv-prim.s12, uv-prim.s12)*f.x + (f.y/K)));
            }
        }
      
        // synchronise
        barrier(CLK_LOCAL_MEM_FENCE);

        // have populated the local array
        // now we sum up the elements, migrating the sum to the top of the stack
        int step = 1;
        int grab_id = 2;
        while (step < LOCAL_SIZE) {
            if (!(thread_id%grab_id)) { // assume LOCAL_SIZE is a power of two
                // reduce
                Q[thread_id] += Q[thread_id+step];
            }
            step *= 2;
            grab_id *= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // the sum of all the moments should be at the top of the stack

        if (thread_id == 0) {
            GQ(mi, mj) = Q[0];
        }
    }
    
    return;
}


