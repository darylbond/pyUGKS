//clUGKS.c

/* The following code is split up into four parts to ensure global 
 * synchronisation of memory read/write. Without this splitting
 * the memory access collide and values do not get written to Fstar
 * 
 * Pattern is as follows:
 * 	perform all South boundary fluxes on odd, then even
 * 	perfrom all West boundary fluxes on odd, then even
 */


__kernel void
UGKS_flux(__global double2* Fin,
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

    size_t gi, gj, gv, thread_id;

    thread_id = get_local_id(2);

    gi = (1-face)*(get_global_id(0)) + face*(2*get_global_id(0) + even_odd) + GHOST;
    gj = (1-face)*(2*get_global_id(1) + even_odd) + face*(get_global_id(1)) + GHOST;

    for (size_t loop_id = 0; loop_id < LOCAL_LOOP_LENGTH; ++loop_id) {

        gv = loop_id*LOCAL_SIZE + thread_id;

        if (gv >= NV) {
            continue;
        }

        // the normal and tangential velocity components relative to the edge
        double2 uv = interfaceVelocity(gi, gj, gv, face, normal);

        // ---< STEP 1 >---
        // reconstruct the initial distribution at the face

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
        //#pragma unroll
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

        // the interface value of f
        __local double2 face_dist[NV];
        face_dist[gv] = f_stencil[MID_STENCIL] + sigma*interface_distance;
    }
  
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // we now have the interface distribution 
  
    
    for (size_t loop_id = 0; loop_id < LOCAL_LOOP_LENGTH; ++loop_id) {

        gv = loop_id*LOCAL_SIZE + thread_id;

        if (gv >= NV) {
            continue;
        }
        
        uv = interfaceVelocity(gi, gj, gv, face, normal);
        
        // ---< STEP 2 >---
        // calculate the macroscopic variables in the local frame of reference
        
        double rho = 0.0;
        double T = 0.0;
        double2 UV = 0.0;
        
        // conserved variables
        for (size_t v_id = 0; v_id < NV; ++v_id) {
            rho += f.x;
            UV += uv*f.x;
            T += 0.5*(dot(uv,uv)*f.x + f.y);
        }
        // convert to standard variables
        T = 0.5*rho/(gam-1.0)/(T - 0.5*dot(UV,UV)/rho);
        UV /= rho;
        
        double2 Q = 0.0;
        for (size_t v_id = 0; v_id < NV; ++v_id) {
            Q.x += 0.5*((uv.x-UV.x)*dot(uv-UV, uv-UV)*f.x + (uv.x-UV.x)*f.y);
            Q.y += 0.5*((uv.y-UV.y)*dot(uv-UV, uv-UV)*f.x + (uv.y-UV.y)*f.y);
        }
        
        // we now have all the macroscopic variables
        
        // ---< STEP 3 >---
        // calculate a^L and a^R
        
    }
    
  /*
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
*/
  return;
}
