
// clRK.cxx

////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK_FLUXES
////////////////////////////////////////////////////////////////////////////////
/* The following code is split up into four parts to ensure global 
 * synchronisation of memory read/write. Without this splitting
 * the memory access collide and values do not get written to Fstar
 * 
 * Pattern is as follows:
 * 	perform all South boundary fluxes on odd, then even
 * 	perfrom all West boundary fluxes on odd, then even
 */

__kernel void
RK_FLUXES(__global double2* Fin,
	   __global double2* flux,
	   __global double2* centre,
	   __global double2* mid_side,
	   __global double2* normal,
	   __global double* side_length,
	   __global double* area,
	   int face, int even_odd,
     __global int* flag)
{
  
  // EVEN: even_odd = 0
  // ODD: even_odd = 1
  
  // global index
  
  size_t ti, tj, gi, gj, gv;
  
  ti = get_local_id(0);
  tj = get_local_id(1);
  
  gi = (1-face)*(get_global_id(0)) + face*(2*get_global_id(0) + even_odd) + GHOST;
  gj = (1-face)*(2*get_global_id(1) + even_odd) + face*(get_global_id(1)) + GHOST;
  gv = get_global_id(2);
  
  // the normal and tangential velocity components relative to the edge
  double2 uv = interfaceVelocity(gi, gj, gv, face, normal);
  
  // now make a stencil for the incoming flow into this cell
  // the stencil will be upwinding
  //  flow direction : -->
  //  +--+-|-+ : van Leer
  //  +--+--+-|-+--+ : WENO5
  
  int2 direction = sign(uv.x);
  direction.x = abs((direction.x + 1)/2); // -> 0 when u < 0
  direction.y = abs((direction.y - 1)/2); // -> 0 when u > 0
  
  double2 c_stencil[STENCIL_LENGTH];
  double2 f_stencil[STENCIL_LENGTH];
  
  int2 sij;
  int offset = 0;
  #pragma unroll
  for (int si = 0; si < STENCIL_LENGTH; si++) {
    if (face == SOUTH) {
      sij.x = gi;
      sij.y = (direction.x)*(gj - MID_STENCIL - 1 + offset);  // into cell
      sij.y += (direction.y)*(gj + MID_STENCIL - offset); // out of cell
    } else if (face == WEST) {
      sij.x = (direction.x)*(gi - MID_STENCIL - 1 + offset);  // into cell
      sij.x += (direction.y)*(gi + MID_STENCIL - offset); // out of cell
      sij.y = gj;
    }
    offset += 1;
    
    // the stencils
    f_stencil[si] = F(sij.x, sij.y, gv);
    c_stencil[si] = CENTRE(sij.x,sij.y);
  }
  
  // made the stencil, now reconstruct the interface distribution
  
  // distance from the middle of the interface to the centre of the upwind cell
  double interface_distance = length(MIDSIDE(gi,gj,face) - c_stencil[MID_STENCIL]);
  
  #if FLUX_METHOD == 0 // van Leer
    double2 s1 = (f_stencil[MID_STENCIL] - f_stencil[MID_STENCIL-1])/length(c_stencil[MID_STENCIL] - c_stencil[MID_STENCIL-1]);
    double2 s2 = (f_stencil[MID_STENCIL+1] - f_stencil[MID_STENCIL])/length(c_stencil[MID_STENCIL-1] - c_stencil[MID_STENCIL]);
    double2 sigma = vanLeer(s1,s2);
  #endif
  #if FLUX_METHOD == 1 // WENO5
    // left side of interface
    double2 sigma = WENO5(f_stencil, interface_distance);
  #endif
  
  // the interface flux
  // this flux is the flux INTO the cell (gi,gj) (positive = in, negative = out)
  __local double2 cell_flux[WSI][WSJ][NV];
  cell_flux[ti][tj][gv] = uv.x*(f_stencil[MID_STENCIL] + sigma*interface_distance);
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  #if HAS_DIFFUSE_WALL == 1
  __local double2 wall_flux[WSI][WSJ][NV];
  __local double ratio[WSI][WSJ][1];
  __local double wall_velocity[WSI][WSJ][NV];
  // check for Diffuse boundary, update fL and fR if on the boundary
  #if ((DIFFUSE_NORTH == 1) || (DIFFUSE_SOUTH == 1))
  if ((face == SOUTH) && (gi <= IMAX)) {
    #if DIFFUSE_SOUTH == 1
    if (gj == JMIN){
      // SOUTH face of cell at "SOUTH" of block
      diffuseReflect(cell_flux, wall_flux, ratio, wall_velocity, GSOUTH, uv, FLUX_IN_S, gi, gj);
    }
    #endif
    #if DIFFUSE_NORTH == 1
    if (gj == JMAX+1) {
      // NORTH face of cell at "NORTH" of block
      diffuseReflect(cell_flux, wall_flux, ratio, wall_velocity, GNORTH, uv, FLUX_IN_N, gi, gj);
    }
    #endif
  }
  #endif
  #if (DIFFUSE_EAST == 1) || (DIFFUSE_WEST == 1)
  if ((face == WEST) && (gj <= JMAX)) {
    #if DIFFUSE_WEST == 1
    if (gi == IMIN) {
      // WEST face of cell at "WEST" of block
      diffuseReflect(cell_flux, wall_flux, ratio, wall_velocity, GWEST, uv, FLUX_IN_W, gi, gj);
    }
    #endif
    #if DIFFUSE_EAST == 1
    if (gi == IMAX+1) {
      // EAST face of cell at "EAST" of block
      diffuseReflect(cell_flux, wall_flux, ratio, wall_velocity, GEAST, uv, FLUX_IN_E, gi, gj);
    }
    #endif
  }
  #endif
  #endif
  
  // update FLUX
  double A;
  double interface_length = LENGTH(gi,gj,face);
  if ((face == SOUTH) && (gi <= IMAX)) {
    A = AREA(gi,gj);
    FLUX(gi,gj,gv) += (interface_length/A)*cell_flux[ti][tj][gv];
    A = AREA(gi,gj-1);
    FLUX(gi,gj-1,gv) -= (interface_length/A)*cell_flux[ti][tj][gv];
  }
  else if ((face == WEST) && (gj <= JMAX)) {
    A = AREA(gi,gj);
    FLUX(gi,gj,gv) += (interface_length/A)*cell_flux[ti][tj][gv];
    A = AREA(gi-1,gj);
    FLUX(gi-1,gj,gv) -= (interface_length/A)*cell_flux[ti][tj][gv];
  }

  return;
}

#define F1(i,j,v) f1[NV*NJ*(i) + NV*(j) + (v)]
#define F2(i,j,v) f2[NV*NJ*(i) + NV*(j) + (v)]
#define F3(i,j,v) f3[NV*NJ*(i) + NV*(j) + (v)]

#define FLUXN(i,j,v) fluxn[NV*NJ*(i) + NV*(j) + (v)]
#define FLUX1(i,j,v) flux1[NV*NJ*(i) + NV*(j) + (v)]
#define FLUX2(i,j,v) flux2[NV*NJ*(i) + NV*(j) + (v)]
#define FLUX3(i,j,v) flux3[NV*NJ*(i) + NV*(j) + (v)]

#if TIME_METHOD == 3
////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK3_STEP1
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK3_STEP1(__global double2* Fin,
          __global double2* f1,
          __global double2* flux,
          double DT,
          __global int* flag)
{
  //perform RK3 stepping

  // global index
  size_t gi = get_global_id(0) + GHOST;
  size_t gj = get_global_id(1) + GHOST;
  int gv = get_global_id(2);
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
    
  double rho, T;
  double2 UV, Q;
  
  // load the velocity distribution
  __local double2 fdist[WSI][WSJ][NV];
  fdist[ti][tj][gv] = F(gi, gj, gv);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // calculate the macroscopic properties
  macroPropLocal(fdist, &rho, &UV, &T, &Q);
  
  double tau_n = relaxTime(rho, T); // relaxation time
  
  ///////////////////////
  // STEP 1 - get the post-streaming distribution
  
  double2 fSeq = fS(rho, UV, T, Q, ghV[gv], gv); // Shakhov equilibrium
  
  double2 fluxr0 = FLUX(gi,gj,gv);
  
  F1(gi,gj,gv) = fdist[ti][tj][gv] + DT*fluxr0 - (DT/tau_n)*(fdist[ti][tj][gv] - fSeq);
  
  return;
}

////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK3_STEP2
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK3_STEP2(__global double2* Fin,
	  __global double2* f1,
	  __global double2* flux,
	  __global double2* f2,
	  double DT,
    __global int* flag)
{
  //perform RK3 stepping
  
  // global index
  size_t mi = get_global_id(0);
  size_t mj = get_global_id(1);
  
  size_t gi = mi + GHOST;
  size_t gj = mj + GHOST;
  int gv = get_global_id(2);
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
    
  double rho, T;
  double2 UV, Q;
  
  // load the velocity distribution
  __local double2 f1dist[WSI][WSJ][NV];
  f1dist[ti][tj][gv] = F1(gi, gj, gv);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // calculate the macroscopic properties
  macroPropLocal(f1dist, &rho, &UV, &T, &Q);
  
  double tau_1 = relaxTime(rho, T); // relaxation time
  
  ///////////////////////
  // STEP 1 - get the post-streaming distribution
    
  double2 uv = ghV[gv];
  
  double2 f_n = F(gi,gj,gv);
  
  double2 fluxr1 = FLUX(gi,gj,gv);
  
  double2 fSeq = fS(rho, UV, T, Q, uv, gv); // Shakhov equilibrium
  
  F2(gi,gj,gv) = (3*f_n + f1dist[ti][tj][gv] + DT*fluxr1)/4. - (DT/(4.*tau_1))*(f1dist[ti][tj][gv] - fSeq);
  
  return;
}


////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK3_UPDATE
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK3_UPDATE(__global double2* Fin,
	   __global double2* f1,
	   __global double2* f2,
	   __global double2* flux,
	   double DT,
     __global int* flag)
{
  //perform RK3 stepping

  // global index
  int mi = get_global_id(0);
  int mj = get_global_id(1);
  
  int gi = mi + GHOST;
  int gj = mj + GHOST;
  int gv = get_global_id(2);
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
  
  double rho, T;
  double2 UV, Q;
  
  // load the velocity distribution
  __local double2 f2dist[WSI][WSJ][NV];
  f2dist[ti][tj][gv] = F2(gi, gj, gv);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // needed for every thread?
  macroPropLocal(f2dist, &rho, &UV, &T, &Q);
  
  double tau_2 = relaxTime(rho, T); // relaxation time
  
  ///////////////////////
  // STEP 1 - get the post-streaming distribution
    
  double2 uv = ghV[gv];
  
  double2 f_n = F(gi,gj,gv);
  
  double2 fSeq = fS(rho, UV, T, Q, uv, gv); // Shakhov equilibrium
  
  double2 fluxr2 = FLUX(gi,gj,gv);
  
  double2 f_new = (f_n + 2*f2dist[ti][tj][gv] + 2*DT*fluxr2)/3. - ((2*DT)/(3.*tau_2))*(f2dist[ti][tj][gv] - fSeq);
  
  F(gi,gj,gv) = f_new;
  
  if (any(isnan(f_new))) {
    flag[0] = 1;
    return;
  }
      
  return;
}

#endif
#if TIME_METHOD == 4
////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK4_STEP1
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK4_STEP1(__global double2* Fin,
	  __global double2* f1,
	  __global double2* fluxn,
	  double DT,
    __global int* flag)
{
  //perform RK4 stepping

  // global index
  size_t gi = get_global_id(0) + GHOST;
  size_t gj = get_global_id(1) + GHOST;
  int gv = get_global_id(2);
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
    
  double rho, T;
  double2 UV, Q;
  
  // load the velocity distribution
  __local double2 fdist[WSI][WSJ][NV];
  fdist[ti][tj][gv] = F(gi, gj, gv);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // calculate the macroscopic properties
  macroPropLocal(fdist, &rho, &UV, &T, &Q);
  
  double tau_n = relaxTime(rho, T); // relaxation time
    
  double2 fSeq = fS(rho, UV, T, Q, ghV[gv], gv); // Shakhov equilibrium
    
  double2 fluxr0 = FLUXN(gi,gj,gv);
  
  F1(gi,gj,gv) = fdist[ti][tj][gv] + (DT/2.0)*fluxr0 - (DT/(2.0*tau_n))*(fdist[ti][tj][gv] - fSeq);

  return;
}

////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK4_STEP2
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK4_STEP2(__global double2* Fin,
	  __global double2* f1,
	  __global double2* flux1,
	  __global double2* f2,
	  double DT,
    __global int* flag)
{
  //perform RK4 stepping
  
  // global index
  size_t gi = get_global_id(0) + GHOST;
  size_t gj = get_global_id(1) + GHOST;
  int gv = get_global_id(2);
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
    
  double rho, T;
  double2 UV, Q;
  
  // load the velocity distribution
  __local double2 f1dist[WSI][WSJ][NV];
  f1dist[ti][tj][gv] = F1(gi, gj, gv);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // calculate the macroscopic properties
  macroPropLocal(f1dist, &rho, &UV, &T, &Q);
  
  double tau_1 = relaxTime(rho, T); // relaxation time
    
  double2 fSeq = fS(rho, UV, T, Q, ghV[gv], gv); // Shakhov equilibrium
  
  double2 f_n = F(gi,gj,gv);
  
  double2 fluxr1 = FLUX1(gi,gj,gv);
  
  F2(gi,gj,gv) = f_n + (DT/2.0)*fluxr1 - (DT/(2.0*tau_1))*(f1dist[ti][tj][gv] - fSeq);

  return;
}

////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK4_STEP3
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK4_STEP3(__global double2* Fin,
	  __global double2* f2,
	  __global double2* flux2,
	  __global double2* f3,
	  double DT,
    __global int* flag)
{
  //perform RK4 stepping

  // global index
  size_t gi = get_global_id(0) + GHOST;
  size_t gj = get_global_id(1) + GHOST;
  int gv = get_global_id(2);
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
    
  double rho, T;
  double2 UV, Q;
  
  // load the velocity distribution
  __local double2 f2dist[WSI][WSJ][NV];
  f2dist[ti][tj][gv] = F2(gi, gj, gv);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // calculate the macroscopic properties
  macroPropLocal(f2dist, &rho, &UV, &T, &Q);
  
  double tau_2 = relaxTime(rho, T); // relaxation time

  double2 fSeq = fS(rho, UV, T, Q, ghV[gv], gv); // Shakhov equilibrium
  
  double2 f_n = F(gi,gj,gv);
  
  double2 fluxr2 = FLUX2(gi,gj,gv);
  
  F3(gi,gj,gv) = f_n + DT*fluxr2 - (DT/tau_2)*(f2dist[ti][tj][gv] - fSeq);
  
  return;
}

////////////////////////////////////////////////////////////////////////////////
//KERNEL: RK4_UPDATE
////////////////////////////////////////////////////////////////////////////////

__kernel void
RK4_UPDATE(__global double2* Fin,
	   __global double2* f1,
	   __global double2* f2,
	   __global double2* f3,
	   __global double2* flux3,
	   double DT,
     __global int* flag)
{
  //perform RK4 stepping

  // global index
  size_t gi = get_global_id(0) + GHOST;
  size_t gj = get_global_id(1) + GHOST;
  int gv = get_global_id(2);
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
    
  double rho, T;
  double2 UV, Q;
  
  // load the velocity distribution
  __local double2 f3dist[WSI][WSJ][NV];
  f3dist[ti][tj][gv] = F3(gi, gj, gv);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // calculate the macroscopic properties
  macroPropLocal(f3dist, &rho, &UV, &T, &Q);
  
  double tau_3 = relaxTime(rho, T); // relaxation time

  double2 fSeq = fS(rho, UV, T, Q, ghV[gv], gv); // Shakhov equilibrium
  
  double2 f_n = F(gi,gj,gv);
  double2 f_1 = F1(gi,gj,gv);
  double2 f_2 = F2(gi,gj,gv);
  
  double2 fluxr3 = FLUX3(gi,gj,gv);
  
  double2 f_new = (1.0/3.0)*(-f_n + f_1 + 2.0*f_2 + f3dist[ti][tj][gv]) + (DT/6.0)*fluxr3 - (DT/(6.0*tau_3))*(f3dist[ti][tj][gv] - fSeq);
  
  F(gi,gj,gv) = f_new;
  
  if (any(isnan(f_new))) {
    flag[0] = 1;
    return;
  }
  
  return;
}
#endif
