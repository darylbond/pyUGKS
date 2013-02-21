/////////////////////////////////////////
//MACROS 
/////////////////////////////////////////\n
#define F(i,j,v) Fin[NV*NJ*(i) + NV*(j) + (v)] 
#define FLUX(i,j,v) flux[NV*NJ*(i) + NV*(j) + (v)]
#define TXYZ(i,j) Txyz[(i)*nj + (j)] 
#define XY(i,j) xy[(i)*NY + (j)] 
#define RES(i,j) residual[(i)*nj + (j)]

#define GD(i,j) gD[(i)*nj + (j)]
#define GUV(i,j) gUV[(i)*nj + (j)] 
#define GT(i,j) gT[(i)*nj + (j)] 
#define GQ(i,j) gQ[(i)*nj + (j)] 


#define AREA(i,j) area[(i)*NJ + (j)] 
#define CENTRE(i,j) centre[(i)*NJ + (j)] 
#define MIDSIDE(i,j,face) mid_side[(i)*NJ*2 + (j)*2 + face] 
#define NORMAL(i,j,face) normal[(i)*NJ*2 + (j)*2 + face] 
#define LENGTH(i,j,face) side_length[(i)*NJ*2 + (j)*2 + face] 

#define TSTEP(i,j) time_step[(i)*nj + (j)]

/////////////////////////////////////////
// KERNEL: initialiseToZero
/////////////////////////////////////////

__kernel void
initialiseToZero(__global double2* flux)
{
  // set all values to zero
  
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t v = get_global_id(2);
  
  FLUX(i,j,v) = 0.0;
  
  return;
}

/////////////////////////////////////////
// KERNEL: copyBuffer
/////////////////////////////////////////

__kernel void
copyBuffer(__global double2* Ain, __global double2* Bout)
{
  // copy from buffer A to buffer B
  
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t v = get_global_id(2);
  
  Bout[NV*NJ*(i) + NV*(j) + (v)] = Ain[NV*NJ*(i) + NV*(j) + (v)];
  
  return;
}

/////////////////////////////////////////
// KERNEL: cellResidual
/////////////////////////////////////////

__kernel void
cellResidual(__global double2* Anew, 
    __global double2* Bold,
    __global double* area,
    __global double2* residual)
{
  // residual
  
  size_t mi = get_global_id(0);
  size_t mj = get_global_id(1);
  
  int gi = mi + GHOST;
  int gj = mj + GHOST;
  
  double rho_n, T_n;
  double2 UV_n;

  macroShort(Anew, gi, gj, &rho_n, &UV_n, &T_n);
  
  double rho_o, T_o;
  double2 UV_o;

  macroShort(Bold, gi, gj, &rho_o, &UV_o, &T_o);
  
  double d_rho, d_M, d_T;
  
  d_rho = rho_n - rho_o;
  d_M = rho_n*length(UV_n) - rho_o*length(UV_o);
  d_T = T_n - T_o;
  
  RES(mi,mj) = AREA(gi, gj)*sqrt(d_rho*d_rho + d_M*d_M + d_T*d_T);
  
  return;
}

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
clFindDT(__global double* gD, __global double* gT, __global double* time_step) 
{
  // calculate the time step parameter for this cell
  
  size_t mi = get_global_id(0);
  size_t mj = get_global_id(1);
  
  double tau = relaxTime(GD(mi,mj), GT(mi,mj));
  
  TSTEP(mi,mj) = CFL*tau;
    
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
    uv = ghV[i];
    
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
  Tx = M4/D - U*U;
  Ty = M5/D - V*V;
  Tz = H/D;
  T = (Tx + Ty + Tz)/B;
  
  TXYZ(mi,mj) = (double4)(Tx, Ty, Tz, T);

  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: initFunctions
////////////////////////////////////////////////////////////////////////////////

// KERNEL FUNCTION
__kernel void
initFunctions(__global double2* Fin, 
__global double* gD, __global double2* gUV, 
__global double* gT, __global double2* gQ) 
{
  // initialise functions, but only those that are within the simulation domain, does not include ghost cells
  
  // distribution functions global index
  size_t mi = get_global_id(0);
  size_t mj = get_global_id(1);
  size_t gv = get_global_id(2);

  // macroscopic properties global index
    
    size_t gi = mi + GHOST;
    size_t gj = mj + GHOST;

  // compute equilibrium functions

  double2 uv = ghV[gv];
  F(gi,gj,gv) = fS(GD(mi,mj), GUV(mi,mj), GT(mi,mj), GQ(mi,mj), uv, gv);
  
  return;
}

__kernel void
calcMacro(__global double2* Fin, 
  __global double* gD, __global double2* gUV, 
  __global double* gT, __global double2* gQ,
  __global int* flag) 
{
  // calculate the macroscopic properties
  
  size_t mi = get_global_id(0);
  size_t mj = get_global_id(1);
  
  int gi = mi + GHOST;
  int gj = mj + GHOST;
  
  double rho, T;
  double2 UV, Q;

  macroProp(Fin, gi, gj, &rho, &UV, &T, &Q);
    
  if (isnan(rho)) {
    flag[0] = 2;
    return;
  }

  GD(mi, mj) = rho;
  GUV(mi,mj) = UV;
  GT(mi,mj) = T;
  GQ(mi,mj) = Q;
  
  return;
}
