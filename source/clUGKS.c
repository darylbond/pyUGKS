//clUGKS.c

/* The following code is split up into four parts to ensure global 
 * synchronisation of memory read/write. Without this splitting
 * the memory access collide and values do not get written to flux
 * 
 * Pattern is as follows:
 * 	perform all South boundary fluxes on odd, then even
 * 	perfrom all West boundary fluxes on odd, then even
 */
 

 #define MNUM 7
 #define MTUM 5


void getInterfaceDist(__global double2* Fin, __global double2* centre, __global double2* mid_side, size_t gi, size_t gj, int face, double2 face_normal, 
    __local double2 face_dist[NV], __local double2 face_slope[NV]) {
    // calculate the interface distribution using a choice of limiter

    for (size_t gv = 0; gv < NV; ++gv) {

        // the normal and tangential velocity components relative to the edge
        double2 uv = interfaceVelocity(gv, face_normal);

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
        face_dist[gv] = f_stencil[MID_STENCIL] + sigma*interface_distance;
        face_slope[gv] = sigma;
    }
    
    return;
}
 
double4 getConserved_local(__local double2 f[NV], double2 normal) {
    // calculate the primary variables given a __local list for the 
    // distribution value
    // RELATIVE TO INTERFACE
    
    double4 w = 0.0;
     
    // conserved variables
    for (size_t v_id = 0; v_id < NV; ++v_id) {
        double2 uv = interfaceVelocity(v_id, normal);
        w.s0 += f[v_id].x;
        w.s12 += uv*f[v_id].x;
        w.s3 += 0.5*(dot(uv,uv)*f[v_id].x + f[v_id].y);
    }
    
    return w;
 }
 
double4 getPrimary(double4 w) {
    // convert to primary variables
    w.s3 = 0.5*w.s0/(gam-1.0)/(w.s3 - 0.5*dot(w.s12,w.s12)/w.s0);
    w.s12 /= w.s0;
    return w;
}

double4 getConserved(double4 prim) {
    // convert to conserved variables
    prim.s3 = 0.5*prim.s0/prim.s3/(gam-1.0)+0.5*prim.s0*(dot(prim.s12,prim.s12));
    prim.s12 *= prim.s0;
    return prim;
}
 
double2 getHeatFlux_local(__local double2 f[NV], double2 normal, double4 w) {
    // calculate the heat flux given a __local list for the 
    // distribution value
    // RELATIVE TO INTERFACE
    
    double2 Q = 0.0;
    for (size_t v_id = 0; v_id < NV; ++v_id) {
        double2 uv = interfaceVelocity(v_id, normal);
        Q.x += 0.5*((uv.x-w.s1)*dot(uv-w.s12, uv-w.s12)*f[v_id].x + (uv.x-w.s1)*f[v_id].y);
        Q.y += 0.5*((uv.y-w.s2)*dot(uv-w.s12, uv-w.s12)*f[v_id].x + (uv.y-w.s2)*f[v_id].y);
    }
    
    return Q;
 }
 
double4 getConserved_global(__global double2* Fin, int gi, int gj, double2 normal) {
    // calculate the primary variables given a __global list for the 
    // distribution value
    // RELATIVE TO INTERFACE

    double4 w = 0.0;
    double2 f, uv;
    // conserved variables
    for (size_t v_id = 0; v_id < NV; ++v_id) {
        uv = interfaceVelocity(v_id, normal);
        f = F(gi, gj, v_id);
        w.s0 += f.x;
        w.s12 += uv*f.x;
        w.s3 += 0.5*(dot(uv,uv)*f.x + f.y);
    }
    
    return w;
 }
 
 double2 getHeatFlux_global(__global double2* Fin, int gi, int gj, double2 normal,  double4 w) {
    // calculate the primary variables given a __local list for the 
    // distribution value
    
    double2 f, uv;
    double2 Q = 0.0;
    
    for (size_t v_id = 0; v_id < NV; ++v_id) {
        uv = interfaceVelocity(v_id, normal);
        f = F(gi, gj, v_id);
        Q.x += 0.5*((uv.x-w.s1)*dot(uv-w.s12, uv-w.s12)*f.x + (uv.x-w.s1)*f.y);
        Q.y += 0.5*((uv.y-w.s2)*dot(uv-w.s12, uv-w.s12)*f.x + (uv.y-w.s2)*f.y);
    }

    return Q;
 }

double4 microSlope(double4 prim, double4 sw) {
    // calculate the micro slop of the Maxwellian
    
    double4 micro_slope;
    
    micro_slope.s3 = 4.0*(prim.s3*prim.s3)/(K+2)/prim.s0*(2.0*sw.s3-2.0
                        *prim.s1*sw.s1-2.0*prim.s2*sw.s2+sw.s0
                        *((prim.s1*prim.s1)+(prim.s2*prim.s2)-0.5*(K+2)/prim.s3));
    micro_slope.s2 = 2.0*prim.s3/prim.s0*(sw.s2-prim.s2*sw.s0)-prim.s2*micro_slope.s3;
    micro_slope.s1 = 2.0*prim.s3/prim.s0*(sw.s1-prim.s1*sw.s0)-prim.s1*micro_slope.s3;
    micro_slope.s0 = sw.s0/prim.s0-prim.s1*micro_slope.s1-prim.s2
                        *micro_slope.s2-0.5*((prim.s1*prim.s1)+(prim.s2*
                        prim.s2)+0.5*(K+2)/prim.s3)*micro_slope.s3;
    
    return micro_slope;
}

void momentU(double4 prim, double Mu[MNUM], double Mv[MTUM], double Mxi[3], double Mu_L[MNUM], double Mu_R[MNUM]) {
    // calculate the moments of velocity
    
    // moments of normal velocity
    Mu_L[0] = 0.5*erfc(-sqrt(prim.s3)*prim.s1);
    Mu_L[1] = prim.s1*Mu_L[0] + 0.5*exp(-prim.s3*(prim.s1*prim.s1))/sqrt(PI*prim.s3);
    
    Mu_R[0] = 0.5*erfc(sqrt(prim.s3)*prim.s1);
    Mu_R[1] = prim.s1*Mu_R[0]-0.5*exp(-prim.s3*(prim.s1*prim.s1))/sqrt(PI*prim.s3);

    for (int i = 2; i < MNUM; ++i) {
        Mu_L[i] = prim.s1*Mu_L[i-1] + 0.5*(i-1)*Mu_L[i-2]/prim.s3;
        Mu_R[i] = prim.s1*Mu_R[i-1] + 0.5*(i-1)*Mu_R[i-2]/prim.s3;
    }
    
    for (int i = 0; i < MNUM; ++i) {
        Mu[i] = Mu_L[i]+Mu_R[i];
    }

    // moments of tangential velocity
    Mv[0] = 1.0;
    Mv[1] = prim.s2;

    for (int i = 2; i < MTUM; ++i) {
        Mv[i] = prim.s2*Mv[i-1]+0.5*(i-1)*Mv[i-2]/prim.s3;
    }

    //moments of \xi
    Mxi[0] = 1.0; //<\xi^0>
    Mxi[1] = 0.5*K/prim.s3; //<\xi^2>
    Mxi[2] = (K*K+2.0*K)/(4.0*prim.s3*prim.s3); //<\xi^4>
    
    return;
}

double4 moment_uv(double Mu[MNUM], double Mv[MTUM], double Mxi[3], int alpha, int beta, int delta) {
    // calculate <u^\alpha*v^\beta*\xi^\delta*\psi>
    
    double4 muv;
    
    muv.s0 = Mu[alpha]*Mv[beta]*Mxi[delta/2];
    muv.s1 = Mu[alpha+1]*Mv[beta]*Mxi[delta/2];
    muv.s2 = Mu[alpha]*Mv[beta+1]*Mxi[delta/2];
    muv.s3 = 0.5*(Mu[alpha+2]*Mv[beta]*Mxi[delta/2]+Mu[alpha]*Mv[beta+2]*Mxi[delta/2]+Mu[alpha]*Mv[beta]*Mxi[(delta+2)/2]);
            
    return muv;
}

double4 moment_au(double4 a, double Mu[MNUM], double Mv[MTUM], double Mxi[3], int alpha, int beta) {
    // calculate <a*u^\alpha*v^\beta*\psi>
    
    return a.s0*moment_uv(Mu,Mv,Mxi,alpha+0,beta+0,0)+
            a.s1*moment_uv(Mu,Mv,Mxi,alpha+1,beta+0,0)+
            a.s2*moment_uv(Mu,Mv,Mxi,alpha+0,beta+1,0)+
            0.5*a.s3*moment_uv(Mu,Mv,Mxi,alpha+2,beta+0,0)+
            0.5*a.s3*moment_uv(Mu,Mv,Mxi,alpha+0,beta+2,0)+
            0.5*a.s3*moment_uv(Mu,Mv,Mxi,alpha+0,beta+0,2);
}

__kernel void
UGKS_flux(__global double2* Fin,
	   __global double2* flux_f,
       __global double4* flux_macro,
	   __global double2* centre,
	   __global double2* mid_side,
	   __global double2* normal,
	   __global double* side_length,
	   __global double* area,
	   int face, int even_odd,
     double dt)
{
  
    // EVEN: even_odd = 0
    // ODD: even_odd = 1

    // global index

    size_t gi, gj;

    gi = (1-face)*(get_global_id(0)) + face*(2*get_global_id(0) + even_odd) + GHOST;
    gj = (1-face)*(2*get_global_id(1) + even_odd) + face*(get_global_id(1)) + GHOST;
    
    double2 face_normal = NORMAL(gi,gj,face);
    // the interface value of f and its slope
    __local double2 face_dist[NV], face_slope[NV];

    // ---< STEP 1 >---
    // reconstruct the initial distribution at the face
    
    getInterfaceDist(Fin, centre, mid_side, gi, gj, face, face_normal, face_dist, face_slope);
    
    //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (face == 1)) {
        //int v_id = 0;
        //printf("gv = %d, h_L = %g, h = %g, h_R = %g, sigma = %g\n",v_id, F(gi-1, gj, v_id), face_dist[v_id], F(gi+1, gj, v_id), face_slope[v_id]);
    //}
    
    // we now have the interface distribution in a local array


    // ---< STEP 2 >---
    // calculate the macroscopic variables in the local frame of reference at the interface
    
    double4 w = getConserved_local(face_dist, face_normal);
    double4 prim = getPrimary(w); // convert to primary variables
    
    // ---< STEP 3 >---
    // calculate a^L and a^R
    double4 sw, aL, aR;
    
    // the slope of the primary variables on the left side of the interface
    sw = w - getConserved_global(Fin, gi-face, gj-(1-face), face_normal); // the difference
    sw /= length(MIDSIDE(gi,gj,face) - CENTRE(gi-face, gj-(1-face))); // the length from cell centre to interface
    
    aL = microSlope(prim, sw);
    
    // the slope of the primary variables on the right side of the interface
    sw = getConserved_global(Fin, gi, gj, face_normal) - w; // the difference
    sw /= length(MIDSIDE(gi,gj,face) - CENTRE(gi, gj)); // the length from cell centre to interface
    
    aR = microSlope(prim, sw);
    
    //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (face == 1)) {
        //printf("w = [%v4g]\nwR = [%v4g]\nsw = [%v4g]\naR = [%v4g]\nlength = %g\n",w, getConserved_global(Fin, gi+face, gj+(1-face), face_normal), sw, aR, length(MIDSIDE(gi,gj,face) - CENTRE(gi, gj)));
        //printf("midside = [%v2g], centre = [%v2g]\n\n",MIDSIDE(gi,gj,face), CENTRE(gi+face, gj+(1-face)));
    //}
    
    // ---< STEP 4 >---
    // calculate the time slope of W and A
    double Mu[MNUM], Mv[MTUM], Mxi[3], Mu_L[MNUM], Mu_R[MNUM];
    
    momentU(prim, Mu, Mv, Mxi, Mu_L, Mu_R);
    
    double4 Mau_L, Mau_R, aT;
    
    Mau_L = moment_au(aL,Mu_L,Mv,Mxi,1,0); //<aL*u*\psi>_{>0}
    Mau_R = moment_au(aR,Mu_R,Mv,Mxi,1,0); //<aR*u*\psi>_{<0}

    sw = -prim.s0*(Mau_L+Mau_R); //time slope of W
    aT = microSlope(prim,sw); //calculate A
    
    // ---< STEP 5 >---
    // calculate collision time and some time integration terms
    double tau = relaxTime(prim);
    
    double Mt[5];
    
    Mt[3] = tau*(1.0-exp(-dt/tau));
    Mt[4] = -tau*dt*exp(-dt/tau)+tau*Mt[3];
    Mt[0] = dt-Mt[3];
    Mt[1] = -tau*Mt[0]+Mt[4]; 
    Mt[2] = (dt*dt)/2.0-tau*Mt[0];
    
    // ---< STEP 6 >---
    // calculate the flux of conservative variables related to g0
    double4 Mau_0, Mau_T;
    
    Mau_0 = moment_uv(Mu,Mv,Mxi,1,0,0); //<u*\psi>
    Mau_L = moment_au(aL,Mu_L,Mv,Mxi,2,0); //<aL*u^2*\psi>_{>0}
    Mau_R = moment_au(aR,Mu_R,Mv,Mxi,2,0); //<aR*u^2*\psi>_{<0}
    Mau_T = moment_au(aT,Mu,Mv,Mxi,1,0); //<A*u*\psi>
    
    double4 face_macro_flux = prim.s0*(Mt[0]*Mau_0 + Mt[1]*(Mau_L+Mau_R) + Mt[2]*Mau_T);
    
    //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (face == 1)) {
        //printf("macro_flux_1 = [%0.15v4g]\n",face_macro_flux);
    //}
    
    // ---< STEP 7 >---
    // calculate the flux of conservative variables related to g+ and f0
    
    double2 Q = getHeatFlux_local(face_dist, face_normal, prim);
    
    double2 F0, f0, uv;
    
    
    double4 macro_flux_temp = 0.0;
    
    // macro flux related to g+ and f0
    for (size_t v_id = 0; v_id < NV; ++v_id) {
        uv = interfaceVelocity(v_id, face_normal);
        F0 = fS(prim, Q, uv, fM(prim, uv, v_id));
        face_macro_flux.s0 += Mt[0]*uv.x*F0.x + Mt[3]*uv.x*face_dist[v_id].x - Mt[4]*uv.x*uv.x*face_slope[v_id].x;
        face_macro_flux.s1 += Mt[0]*uv.x*uv.x*F0.x + Mt[3]*uv.x*uv.x*face_dist[v_id].x - Mt[4]*uv.x*uv.x*uv.x*face_slope[v_id].x;
        face_macro_flux.s2 += Mt[0]*uv.y*uv.x*F0.x + Mt[3]*uv.y*uv.x*face_dist[v_id].x - Mt[4]*uv.y*uv.x*uv.x*face_slope[v_id].x;
        face_macro_flux.s3 += Mt[0]*0.5*(uv.x*dot(uv,uv)*F0.x + uv.x*F0.y) + 
                              Mt[3]*0.5*(uv.x*dot(uv,uv)*face_dist[v_id].x + uv.x*face_dist[v_id].y) - 
                              Mt[4]*0.5*(uv.x*uv.x*dot(uv,uv)*face_slope[v_id].x + uv.x*uv.x*face_slope[v_id].y);
    }
    
    //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (face == 1)) {
        //printf("macro_flux_2 = [%0.15v4g]\n",face_macro_flux);
    //}
    
    // convert macro to global frame
    face_macro_flux.s12 = toGlobal(face_macro_flux.s12, face_normal);
    
    //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (face == 1)) {
        //printf("macro_flux_3 = [%0.15v4g]\n",face_macro_flux);
    //}
    
    
    
    double A;
    double interface_length = LENGTH(gi,gj,face);
    
    if ((face == SOUTH) && (gi <= IMAX)) {
        A = AREA(gi,gj);
        FLUXM(gi,gj) += (interface_length/A)*face_macro_flux;
        A = AREA(gi,gj-1);
        FLUXM(gi,gj-1) -= (interface_length/A)*face_macro_flux;
    }
    else if ((face == WEST) && (gj <= JMAX)) {
        A = AREA(gi,gj);
        FLUXM(gi,gj) += (interface_length/A)*face_macro_flux;
        A = AREA(gi-1,gj);
        FLUXM(gi-1,gj) -= (interface_length/A)*face_macro_flux;
    }
    
    //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (face == 1)) {
        //printf("fluxm = [%0.15v4g]\n\n",(interface_length)*face_macro_flux);
    //}
    
    // ---< STEP 8 >---
    // calculate flux of distribution function
    double2 face_flux;
        
    for (size_t gv = 0; gv < NV; ++gv) {
        
        uv = interfaceVelocity(gv, face_normal);
        
        int delta = (sign(uv.x)+1)/2;
        
        f0 = fM(prim, uv, gv);
        F0 = fS(prim, Q, uv, f0);
        
        face_flux.x = Mt[0]*uv.x*(f0.x+F0.x)+
                      Mt[1]*(uv.x*uv.x)*(aL.s0*f0.x+aL.s1*uv.x*f0.x+aL.s2*uv.y*f0.x+0.5*aL.s3*(dot(uv,uv)*f0.x+f0.y))*delta+
                      Mt[1]*(uv.x*uv.x)*(aR.s0*f0.x+aR.s1*uv.x*f0.x+aR.s2*uv.y*f0.x+0.5*aR.s3*(dot(uv,uv)*f0.x+f0.y))*(1-delta)+
                      Mt[2]*uv.x*(aT.s0*f0.x+aT.s1*uv.x*f0.x+aT.s2*uv.y*f0.x+0.5*aT.s3*(dot(uv,uv)*f0.x+f0.y))+
                      Mt[3]*uv.x*face_dist[gv].x-
                      Mt[4]*(uv.x*uv.x)*face_slope[gv].x;
        
        face_flux.y = Mt[0]*uv.x*(f0.y+F0.y)+
                      Mt[1]*(uv.x*uv.x)*(aL.s0*f0.y+aL.s1*uv.x*f0.y+aL.s2*uv.y*f0.y+0.5*aL.s3*(dot(uv,uv)*f0.y+Mxi[2]*f0.x))*delta+
                      Mt[1]*(uv.x*uv.x)*(aR.s0*f0.y+aR.s1*uv.x*f0.y+aR.s2*uv.y*f0.y+0.5*aR.s3*(dot(uv,uv)*f0.y+Mxi[2]*f0.x))*(1-delta)+
                      Mt[2]*uv.x*(aT.s0*f0.y+aT.s1*uv.x*f0.y+aT.s2*uv.y*f0.y+0.5*aT.s3*(dot(uv,uv)*f0.y+Mxi[2]*f0.x))+
                      Mt[3]*uv.x*face_dist[gv].y-
                      Mt[4]*(uv.x*uv.x)*face_slope[gv].y;
                      
        
        //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (gv == 1) && (face == 1)) {
            //printf("dt = %0.15g, tau = %0.15g\n",dt, tau);
            //printf("uv = [%0.15v2g]\n",uv);
            //printf("Mt = [%0.15g, %0.15g, %0.15g, %0.15g, %0.15g]\n",Mt[0], Mt[1], Mt[2], Mt[3], Mt[4]);
            //printf("Mxi = [%0.15g, %0.15g, %0.15g]\n",Mxi[0], Mxi[1], Mxi[2]);
            //printf("f0 = [%0.15v2g], F0 = [%0.15v2g]\n",f0, F0);
            //printf("aL = [%0.15v4g]\naR = [%0.15v4g]\n",aL, aR);
            //printf("aT = [%0.15v4g]\n",aT);
            //printf("face_dist = [%0.15v2g]\n",face_dist[gv]);
            //printf("face_slope = [%0.15v2g]\n",face_slope[gv]);
            //printf("gv = %d, flux_f = [%0.15v2g]\n\n",gv,face_flux);
        //}
    
    
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
  * */
        
        // ---< STEP 9 >---
        // update the global flux counters
        
        // update FLUX
        if ((face == SOUTH) && (gi <= IMAX)) {
            A = AREA(gi,gj);
            FLUXF(gi,gj,gv) += (interface_length/A)*face_flux;
            A = AREA(gi,gj-1);
            FLUXF(gi,gj-1,gv) -= (interface_length/A)*face_flux;
        }
        else if ((face == WEST) && (gj <= JMAX)) {
            A = AREA(gi,gj);
            FLUXF(gi,gj,gv) += (interface_length/A)*face_flux;
            A = AREA(gi-1,gj);
            FLUXF(gi-1,gj,gv) -= (interface_length/A)*face_flux;
        }
        
        //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (gv == 1) && (face == 1)) {
            //printf("fluxf = [%0.15v2g]\n\n",(interface_length)*face_flux);
        //}
    }

  return;
}

#define RES(i,j) residual[(i)*nj + (j)]

__kernel void
UGKS_update(__global double2* Fin,
	   __global double2* flux_f,
       __global double4* flux_macro,
       __global double4* macro,
       double dt)
{
    // update the macro-properties and distribution
    
    size_t mi, mj, gi, gj, gv, thread_id;
    
    mi = get_global_id(0);
    mj = get_global_id(1);
    thread_id = get_local_id(2);
    
    gi = mi + GHOST;
    gj = mj + GHOST;
    
    // old stuff at t_n
    double4 prim_old = MACRO(mi,mj);
    double4 w_old = getConserved(prim_old);
    double tau_old =  relaxTime(prim_old);
    double2 Q = getHeatFlux_global(Fin, gi, gj, (double2)(1.0, 0.0), prim_old); // aligned with global coords
    
    // update the macro variables
    double4 w = w_old + FLUXM(gi, gj);
    double4 prim = getPrimary(w);
    
    
    // new relaxation rate
    double tau = relaxTime(prim);
        
    if (thread_id == 0) {
        MACRO(mi,mj) = prim;
        
        //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN)) {
            //printf("w_old = [%0.15v4g]\n",w_old);
            //printf("prim_old = [%0.15v4g]\n",prim_old);
            //printf("tau_old = [%0.15g]\n",tau_old);
            //printf("Q = [%0.15v2g]\n",Q);
            //printf("fluxm = [%0.15v4g]\n",FLUXM(gi, gj));
            //printf("w = [%0.15v4g]\n",w);
            //printf("prim = [%0.15v4g]\n",prim);
            //printf("tau = %0.15g\n",tau);
        //}
    }
    
    for (size_t loop_id = 0; loop_id < LOCAL_LOOP_LENGTH; ++loop_id) {

        gv = loop_id*LOCAL_SIZE + thread_id;

        if (gv >= NV) {
            continue;
        }
        
        // old equilibrium
        double2 feq_old = fEQ(prim_old, Q, QUAD[gv], gv);
        
        // new stuff at t^n+1
        double2 feq = fEQ(prim, Q, QUAD[gv], gv);
        
        // update the distribution function
        
        double2 f_old = F(gi,gj,gv);
        
        F(gi,gj,gv) = (f_old + FLUXF(gi,gj,gv) + 0.5*dt*(feq/tau+(feq_old-f_old)/tau_old))/(1.0+0.5*dt/tau);
        
        //if ((BLOCK == 1) && (gi == IMIN) && (gj == JMIN) && (gv == 1)) {
            //printf("\nfeq_old = [%0.15v2g]\n",feq_old);
            //printf("feq = [%0.15v2g]\n",feq);
            //printf("f_old = [%0.15v2g]\n",f_old);
            //printf("fluxf = [%0.15v2g]\n",FLUXF(gi,gj,gv));
            //printf("tau = %0.15g\n",tau);
            //printf("f_new = [%0.15v2g]\n\n",F(gi,gj,gv));
        //}
    }
    
    return;
}
