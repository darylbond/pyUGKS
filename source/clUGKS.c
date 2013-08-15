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

double4 microSlope(double4 prim, double4 sw) {
    // calculate the micro slop of the Maxwellian
    
    double4 micro_slope;
    
    micro_slope.s3 = 4.0*(prim.s3*prim.s3)/(K+2)/prim.s0*(2.0*sw.s3-2.0
                        *prim.s1*sw.s1-2.0*prim.s2*sw.s2+sw.s0
                        *(dot(prim.s12,prim.s12)-0.5*(K+2)/prim.s3));
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

#define IFACEF(i,j,v) iface_f[NV*NJ*(i) + NV*(j) + (v)]
#define FSIGMA(i,j,v) fsigma[NV*NJ*(i) + NV*(j) + (v)]


__kernel void
iFace(__global double2* Fin,
	   __global double2* iface_f,
       __global double2* fsigma,
       __global double2* centre,
	   __global double2* mid_side,
	   __global double2* normal,
       int face)
{
    // get the interface distribution, load it into global memory

    size_t mi = get_global_id(0);
    size_t mj = get_global_id(1);

    if ((((face == SOUTH) && (mi < ni)) && (mj < nj+1)) 
        || (((face == WEST) && (mi < ni+1)) && (mj < nj))) {

        size_t gi = get_global_id(0) + GHOST;
        size_t gj = get_global_id(1) + GHOST;

        double2 face_normal = NORMAL(gi,gj,face);

        size_t thread_id = get_local_id(2);

        for (size_t loop_id = 0; loop_id < LOCAL_LOOP_LENGTH; ++loop_id) {

            size_t gv = loop_id*LOCAL_SIZE + thread_id;

            if (gv < NV) {
                // the normal and tangential velocity components relative to the edge
                double2 uv = interfaceVelocity(gv, face_normal);

                // now make a stencil for the incoming flow into this cell
                // the stencil will be upwinding
                //  flow direction : -->
                //  +--+-|-+ : van Leer
                //  +--+--+-|-+--+ : WENO5
                
                int delta = (sign(uv.x)+1)/2;

                double2 c_stencil[STENCIL_LENGTH];
                double2 f_stencil[STENCIL_LENGTH];

                int2 sij;
                int offset = 0;
                //#pragma unroll
                for (int si = 0; si < STENCIL_LENGTH; si++) {
                    if (face == SOUTH) {
                        sij.x = gi;
                        sij.y = delta*(gj - MID_STENCIL - 1 + offset);  // into cell
                        sij.y += (1-delta)*(gj + MID_STENCIL - offset); // out of cell
                    } else if (face == WEST) {
                        sij.x = delta*(gi - MID_STENCIL - 1 + offset);  // into cell
                        sij.x += (1-delta)*(gi + MID_STENCIL - offset); // out of cell
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
                IFACEF(gi,gj,gv) = f_stencil[MID_STENCIL] + sigma*interface_distance;
                FSIGMA(gi,gj,gv) = sigma;
            }
        }
    }
    return;
}

#define FLUXF(i,j,v) flux_f[NV*NJ*(i) + NV*(j) + (v)]
#define FLUXM(i,j) flux_macro[(i)*NJ + (j)] 

#define PRIM(i,j) primary[(i)*NJ + (j)]
#define AL(i,j) gaL[(i)*NJ + (j)]
#define AR(i,j) gaR[(i)*NJ + (j)]
#define AT(i,j) gaT[(i)*NJ + (j)]

__kernel void
getFaceCons(__global double2* iface_f,
	         __global double2* normal,
             int face,
             __global double4* primary,
             int offset_bottom, int offset_top)
{
  size_t mi, mj, gi, gj, thread_id;
    
    mi = get_global_id(0) + face*offset_bottom;
    mj = get_global_id(1) + (1-face)*offset_bottom;
    thread_id = get_local_id(2);
    
    __local double4 P[LOCAL_SIZE];
    
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
    || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
    
        gi = mi + GHOST;
        gj = mj + GHOST;
        
        // initialise
        P[thread_id] = 0.0;
        
        double2 f, uv;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                f = IFACEF(gi,gj,gv);
                uv = interfaceVelocity(gv, NORMAL(gi,gj,face));
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
            PRIM(gi,gj) = P[0];
        }
    }
    
    return;
}

__kernel void
getAL(__global double2* Fin,
	         __global double2* normal,
             int face,
             __global double4* gaL,
             int offset_bottom, int offset_top)
{
  size_t mi, mj, gi, gj, thread_id;
    
    mi = get_global_id(0) + face*offset_bottom;
    mj = get_global_id(1) + (1-face)*offset_bottom;
    thread_id = get_local_id(2);
    
    __local double4 P[LOCAL_SIZE];
    
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
    || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
    
        gi = mi + GHOST;
        gj = mj + GHOST;
        
        // initialise
        P[thread_id] = 0.0;
        
        double2 f, uv;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                f = F(gi-face, gj-(1-face), gv);
                uv = interfaceVelocity(gv, NORMAL(gi,gj,face));
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
            AL(gi,gj) = P[0];
        }
    }
    
    return;
}

__kernel void
getAR(__global double2* Fin,
	         __global double2* normal,
             int face,
             __global double4* gaR,
             int offset_bottom, int offset_top)
{
  size_t mi, mj, gi, gj, thread_id;
    
    mi = get_global_id(0) + face*offset_bottom;
    mj = get_global_id(1) + (1-face)*offset_bottom;
    thread_id = get_local_id(2);
    
    __local double4 P[LOCAL_SIZE];
    
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
    || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
    
        gi = mi + GHOST;
        gj = mj + GHOST;
        
        // initialise
        P[thread_id] = 0.0;
        
        double2 f, uv;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                f = F(gi, gj, gv);
                uv = interfaceVelocity(gv, NORMAL(gi,gj,face));
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
            AR(gi,gj) = P[0];
        }
    }
    
    return;
}

__kernel void
getPLR(__global double2* centre,
        __global double2* mid_side,
        int face,
        __global double4* primary,
        __global double4* gaL,
        __global double4* gaR,
        int offset_bottom, int offset_top)
{
  size_t mi, mj, gi, gj;
    
    mi = get_global_id(0) + face*offset_bottom;
    mj = get_global_id(1) + (1-face)*offset_bottom;
    
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
    || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
    
        gi = mi + GHOST;
        gj = mj + GHOST;
        
        double4 w = PRIM(gi,gj);        
        double4 prim = getPrimary(w); // convert to primary variables
        
        double4 sw;
        double2 midside = MIDSIDE(gi,gj,face);
        
        // LEFT
        sw = (w - AL(gi,gj))/length(midside - CENTRE(gi-face, gj-(1-face)));
        
        AL(gi,gj) = microSlope(prim, sw);
        
        // RIGHT
        
        sw = (AR(gi,gj) - w)/length(midside - CENTRE(gi,gj));
        
        AR(gi,gj) = microSlope(prim, sw);
        
        PRIM(gi,gj) = prim;
    }

  return;
}

#define MXI(i,j,k) gMxi[3*NJ*(i) + 3*(j) + (k)]

__kernel void
initMacroFlux(__global double4* flux_macro,
	   int face, double dt,
       __global double4* primary,
       __global double4* gaL,
       __global double4* gaR,
       __global double4* gaT,
       __global double* gMxi,
       int offset_bottom, int offset_top)
{
    // global index
    
    size_t mi, mj, gi, gj;
    
    mi = get_global_id(0) + face*offset_bottom;
    mj = get_global_id(1) + (1-face)*offset_bottom;
    
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
    || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
    
        gi = mi + GHOST;
        gj = mj + GHOST;
        
        double4 prim = PRIM(gi,gj);
        double4 aL = AL(gi,gj);
        double4 aR = AR(gi,gj);
        
        // ---< STEP 4 >---
        // calculate the time slope of W and A
        double Mu[MNUM], Mv[MTUM], Mxi[3], Mu_L[MNUM], Mu_R[MNUM];
        
        momentU(prim, Mu, Mv, Mxi, Mu_L, Mu_R);
        
        MXI(gi,gj,0) = Mxi[0];
        MXI(gi,gj,1) = Mxi[1];
        MXI(gi,gj,2) = Mxi[2];
        
        double4 Mau_L, Mau_R, aT, sw;
        
        Mau_L = moment_au(aL,Mu_L,Mv,Mxi,1,0); //<aL*u*\psi>_{>0}
        Mau_R = moment_au(aR,Mu_R,Mv,Mxi,1,0); //<aR*u*\psi>_{<0}

        sw = -prim.s0*(Mau_L+Mau_R); //time slope of W
        aT = microSlope(prim,sw); //calculate A
        
        AT(gi, gj) = aT;
        
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
        
        FLUXM(gi,gj) = prim.s0*(Mt[0]*Mau_0 + Mt[1]*(Mau_L+Mau_R) + Mt[2]*Mau_T);
    }

  return;
}

#define FACEQ(i,j) faceQ[(i)*NJ + (j)]

__kernel void
calcFaceQ(__global double2* iface_f, 
            __global double4* primary, 
            __global double2* normal,
            __global double2* faceQ, int face,
            int offset_bottom, int offset_top) 
{
  // calculate the macroscopic properties

    size_t mi = get_global_id(0) + face*offset_bottom;
    size_t mj = get_global_id(1) + (1-face)*offset_bottom;
    size_t thread_id = get_local_id(2);
    
    __local double2 Q[LOCAL_SIZE];
  
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
        || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
  
        int gi = mi + GHOST;
        int gj = mj + GHOST;
        
        // initialise
        Q[thread_id] = 0.0;
        
        double2 face_normal = NORMAL(gi,gj,face);
        double4 prim = PRIM(gi,gj);
        double2 f, uv;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                f = IFACEF(gi,gj,gv);
                uv = interfaceVelocity(gv, face_normal);
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
            FACEQ(gi, gj) = Q[0];
        }
    }
    
    return;
}

__kernel void
macroFlux(__global double2* flux_f,
       __global double2* fsigma,
       __global double4* flux_macro,
	   __global double2* normal,
	   __global double* side_length,
	   int face, double dt,
       __global double4* primary,
       __global double2* faceQ,
       int offset_bottom, int offset_top)
{
    // global index
    
    size_t mi, mj, gi, gj, thread_id;
    
    mi = get_global_id(0) + face*offset_bottom;
    mj = get_global_id(1) + (1-face)*offset_bottom;
    thread_id = get_local_id(2);
    
    __local double4 face_macro_flux[LOCAL_SIZE];
    
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
    || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
    
        gi = mi + GHOST;
        gj = mj + GHOST;
        
        double2 face_normal = NORMAL(gi,gj,face);
        
        double4 prim = PRIM(gi,gj);
        
        double tau = relaxTime(prim);
        
        double Mt[5];
        
        Mt[3] = tau*(1.0-exp(-dt/tau));
        Mt[4] = -tau*dt*exp(-dt/tau)+tau*Mt[3];
        Mt[0] = dt-Mt[3];
        Mt[1] = -tau*Mt[0]+Mt[4]; 
        Mt[2] = (dt*dt)/2.0-tau*Mt[0];
        
        face_macro_flux[thread_id] = 0.0;
        
        double2 Q = FACEQ(gi,gj);
        
        double2 F0, uv, face_dist, face_slope;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t v_id = li*LOCAL_SIZE+thread_id;
            if (v_id < NV) {
                uv = interfaceVelocity(v_id, face_normal);
                F0 = fS(prim, Q, uv, fM(prim, uv, v_id));
                face_dist = FLUXF(gi,gj,v_id);
                face_slope = FSIGMA(gi,gj,v_id);
                
                face_macro_flux[thread_id].s0 += Mt[0]*uv.x*F0.x + Mt[3]*uv.x*face_dist.x - Mt[4]*uv.x*uv.x*face_slope.x;
                face_macro_flux[thread_id].s1 += Mt[0]*uv.x*uv.x*F0.x + Mt[3]*uv.x*uv.x*face_dist.x - Mt[4]*uv.x*uv.x*uv.x*face_slope.x;
                face_macro_flux[thread_id].s2 += Mt[0]*uv.y*uv.x*F0.x + Mt[3]*uv.y*uv.x*face_dist.x - Mt[4]*uv.y*uv.x*uv.x*face_slope.x;
                face_macro_flux[thread_id].s3 += Mt[0]*0.5*(uv.x*dot(uv,uv)*F0.x + uv.x*F0.y) + 
                                                 Mt[3]*0.5*(uv.x*dot(uv,uv)*face_dist.x + uv.x*face_dist.y) - 
                                                 Mt[4]*0.5*(uv.x*uv.x*dot(uv,uv)*face_slope.x + uv.x*uv.x*face_slope.y);
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
                face_macro_flux[thread_id] += face_macro_flux[thread_id+step];
            }
            step *= 2;
            grab_id *= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (thread_id == 0) {
            // convert macro to global frame
            double4 out = face_macro_flux[0];
            
            out += FLUXM(gi,gj);
            
            out.s12 = toGlobal(out.s12, face_normal);
            
            FLUXM(gi,gj) = LENGTH(gi,gj,face)*out;
        }
    }

  return;
}

__kernel void
distFlux(__global double2* flux_f,
       __global double2* fsigma,
	   __global double2* normal,
	   __global double* side_length,
	   int face, double dt,
       __global double4* primary,
       __global double4* gaL,
       __global double4* gaR,
       __global double4* gaT,
       __global double* gMxi,
       __global double2* faceQ,
       int offset_bottom, int offset_top)
{
    // global index
    
    size_t mi, mj, gi, gj, thread_id;
    
    mi = get_global_id(0) + face*offset_bottom;
    mj = get_global_id(1) + (1-face)*offset_bottom;
    thread_id = get_local_id(2);
    
    if ((((face == SOUTH) && (mi < ni)) && (mj < (nj+1-offset_top))) 
    || (((face == WEST) && (mi < (ni+1-offset_top))) && (mj < nj))) {
    
        gi = mi + GHOST;
        gj = mj + GHOST;
        
        double4 prim = PRIM(gi,gj);
        double4 aL = AL(gi,gj);
        double4 aR = AR(gi,gj);
        double4 aT = AT(gi,gj);
        
        double Mxi[3];
        Mxi[0] = MXI(gi,gj,0);
        Mxi[1] = MXI(gi,gj,1);
        Mxi[2] = MXI(gi,gj,2);
        
        double tau = relaxTime(prim);
        
        double Mt[5];
        
        Mt[3] = tau*(1.0-exp(-dt/tau));
        Mt[4] = -tau*dt*exp(-dt/tau)+tau*Mt[3];
        Mt[0] = dt-Mt[3];
        Mt[1] = -tau*Mt[0]+Mt[4]; 
        Mt[2] = (dt*dt)/2.0-tau*Mt[0];
        
        double2 face_normal = NORMAL(gi,gj,face);
        
        double2 Q = FACEQ(gi,gj);
        
        double2 face_flux, uv, f0, F0;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
            
                uv = interfaceVelocity(gv, face_normal);
                
                int delta = (sign(uv.x)+1)/2;
                
                f0 = fM(prim, uv, gv);
                F0 = fS(prim, Q, uv, f0);
                
                double2 face_dist = FLUXF(gi,gj,gv);
                double2 face_slope = FSIGMA(gi,gj,gv);
                
                face_flux.x = Mt[0]*uv.x*(f0.x+F0.x)+
                                  Mt[1]*(uv.x*uv.x)*(aL.s0*f0.x+aL.s1*uv.x*f0.x+aL.s2*uv.y*f0.x+0.5*aL.s3*(dot(uv,uv)*f0.x+f0.y))*delta+
                                  Mt[1]*(uv.x*uv.x)*(aR.s0*f0.x+aR.s1*uv.x*f0.x+aR.s2*uv.y*f0.x+0.5*aR.s3*(dot(uv,uv)*f0.x+f0.y))*(1-delta)+
                                  Mt[2]*uv.x*(aT.s0*f0.x+aT.s1*uv.x*f0.x+aT.s2*uv.y*f0.x+0.5*aT.s3*(dot(uv,uv)*f0.x+f0.y))+
                                  Mt[3]*uv.x*face_dist.x-
                                  Mt[4]*(uv.x*uv.x)*face_slope.x;
                
                face_flux.y = Mt[0]*uv.x*(f0.y+F0.y)+
                                  Mt[1]*(uv.x*uv.x)*(aL.s0*f0.y+aL.s1*uv.x*f0.y+aL.s2*uv.y*f0.y+0.5*aL.s3*(dot(uv,uv)*f0.y+Mxi[2]*f0.x))*delta+
                                  Mt[1]*(uv.x*uv.x)*(aR.s0*f0.y+aR.s1*uv.x*f0.y+aR.s2*uv.y*f0.y+0.5*aR.s3*(dot(uv,uv)*f0.y+Mxi[2]*f0.x))*(1-delta)+
                                  Mt[2]*uv.x*(aT.s0*f0.y+aT.s1*uv.x*f0.y+aT.s2*uv.y*f0.y+0.5*aT.s3*(dot(uv,uv)*f0.y+Mxi[2]*f0.x))+
                                  Mt[3]*uv.x*face_dist.y-
                                  Mt[4]*(uv.x*uv.x)*face_slope.y;
                
                // update FLUX
                FLUXF(gi,gj,gv) = LENGTH(gi,gj,face)*face_flux;
            }
        }
    }

  return;
}

#if HAS_ACCOMMODATING_WALL == 1  // diffuse 
// the kernel here has been split to accomodate the vagaries of GPU computing
__kernel void
accommodatingWallDist(__global double2* normal,
            int face,
            __global double4* wall_prop,
            __global double2* flux_f, 
            double dt)
{
    // given the flux information for each wall of each cell
    // modify the fluxes at the defined wall to give a diffuse wall
    

    size_t gi = get_global_id(0);
    size_t gj = get_global_id(1);
    size_t thread_id = get_local_id(2);
    size_t ci;
    
    int rot;
    int face_id;

    switch (face) {
        case GNORTH:
            ci = gi;
            gi += GHOST;
            gj += NJ - GHOST;
            rot = -1;
            face_id = SOUTH;
            break;
        case GEAST:
            ci = gj;
            gi += NI - GHOST;
            gj += GHOST;
            rot = -1;
            face_id = WEST;
            break;
        case GSOUTH:
            ci = gi;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = SOUTH;
            break;
        case GWEST:
            ci = gj;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = WEST;
            break;
    }
    
    __local double2 data[LOCAL_SIZE];

    if (((face_id == SOUTH) && ((gi - GHOST) < ni)) 
    || ((face_id == WEST) && ((gj - GHOST) < nj))) {

        // get the interface distribution and the flux out due to this distribution

        double2 face_normal = NORMAL(gi,gj,face_id);
        
        double4 wall = WALL_PROP(face, ci);
        wall.s12 = toLocal(wall.s12, face_normal);
        
        double2 uv, face_dist, wall_dist;
        int delta;
        
        //data = [sum_out, sum_in, 0, 0] (x,y)
        
        data[thread_id] = 0.0;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                uv = interfaceVelocity(gv, face_normal);

                face_dist = FLUXF(gi,gj,gv);

                delta = (sign(uv.x)*rot + 1)/2;

                data[thread_id].x += uv.x*(1-delta)*face_dist.x;
                
                wall_dist = fM(wall, uv, gv);

                data[thread_id].y -= uv.x*delta*wall_dist.x;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // perform reduction
        
        int step = 1;
        int grab_id = 2;
        while (step < LOCAL_SIZE) {
            if (!(thread_id%grab_id)) { // assume LOCAL_SIZE is a power of two
                // reduce
                data[thread_id] += data[thread_id+step];
            }
            step *= 2;
            grab_id *= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }


        double ratio = data[0].x/data[0].y;
        
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculate the flux that would come back in if an equilibrium distribution resided in the wall
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                uv = interfaceVelocity(gv, face_normal);
                delta = (sign(uv.x)*rot + 1)/2;
                
                face_dist = FLUXF(gi,gj,gv);
                
                wall_dist = ratio*delta*fM(wall, uv, gv) + (1-delta)*face_dist;
                
                FLUXF(gi,gj,gv) = wall_dist;
                
                //~ if (((face == GNORTH) && (ci == 0))){
                    //~ printf("gv = %d, wall_dist = [%v2g]\n",gv, wall_dist);
                //~ }
            }
        }
    }

    return;

}

#endif

#if HAS_ACCOMMODATING_WALL > 1  // adsorb_CL & adsorb_specular-diffuse
#define COVER(w, i) wall_cover[(i)*4 + (w)]
#define WALL_DIST(i, v) wall_dist[(i)*NV + (v)]

__kernel void
adsorbingWall_P1(__global double2* normal,
                    int face,
                    __global double4* wall_prop,
                    __global double4* wall_cover,
                    __global double2* wall_dist,
                    __global double2* flux_f,
                    __global double4* macro,
                    double dt)
{
    // Calculate the incoming velocity distribution that is to be 
    // reflected from the wall. 
    
    size_t gi = get_global_id(0);
    size_t gj = get_global_id(1);
    size_t thread_id = get_local_id(2);
    size_t ci;
    
    int rot;
    int face_id;
    
    // mean temperature of cell next to wall, should _really_ be 
    // using temp of face_dist but this would be expensive to calculate
    double T; 

    switch (face) {
        case GNORTH:
            ci = gi;
            gi += GHOST;
            gj += NJ - GHOST;
            rot = -1;
            face_id = SOUTH;
            T = 1.0/MACRO(ci, gj-GHOST-1).s3;
            break;
        case GEAST:
            ci = gj;
            gi += NI - GHOST;
            gj += GHOST;
            rot = -1;
            face_id = WEST;
            T = 1.0/MACRO(gi-GHOST-1, ci).s3;
            break;
        case GSOUTH:
            ci = gi;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = SOUTH;
            T = 1.0/MACRO(ci, gj-GHOST).s3;
            break;
        case GWEST:
            ci = gj;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = WEST;
            T = 1.0/MACRO(gi-GHOST, ci).s3;
            break;
    }
    
    __local double2 data[LOCAL_SIZE];
    __local double data2[LOCAL_SIZE];

    if (((face_id == SOUTH) && ((gi - GHOST) < ni)) 
    || ((face_id == WEST) && ((gj - GHOST) < nj))) {

        // calculate the updated wall coverage

        double2 face_normal = NORMAL(gi,gj,face_id);
        
        double beta;
        double2 total_flux, reflected_flux, adsorbed_flux, desorbed_flux;
        double2 uv, face_dist;
        int delta;
        
        //data = [reflected flux, adsorbed flux]
        
        data[thread_id] = 0.0;
        data2[thread_id] = 0.0;
        
        double vartheta = COVER(face, ci).s0; // the fraction of cover that this wall section has
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                uv = interfaceVelocity(gv, face_normal);

                face_dist = FLUXF(gi,gj,gv); // dist. on the interface

                delta = (sign(uv.x)*rot + 1)/2; // flag to indicate if this velocity is going out of the wall (delta = 1 -> out of wall)
                
                total_flux = -rot*uv.x*(1-delta)*face_dist; // total flux that is impinging on wall
                
                beta = stickingProbability(uv, T); // probability giving the amount of this discrete velocity that has a chance of sticking 
                
                adsorbed_flux = beta*total_flux*GAMMA_F*(1.0 - vartheta); // is adsorbed
                
                if (adsorbed_flux.x > total_flux.x) {
                    adsorbed_flux = total_flux;
                }
                
                reflected_flux = total_flux - adsorbed_flux; // is reflected
                
                
                // sum of the reflected and adsorbed distributions
                data[thread_id].x += reflected_flux.x;
                data[thread_id].y += adsorbed_flux.x;
                
                // the flux that is seen by the Langmuir reaction
                data2[thread_id] += beta*total_flux.x;
                
                WALL_DIST(ci, gv) = reflected_flux/uv.x;  // save the reflected distribution for later
                
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // perform reduction
        int step = 1;
        int grab_id = 2;
        while (step < LOCAL_SIZE) {
            if (!(thread_id%grab_id)) { // assume LOCAL_SIZE is a power of two
                // reduce
                data[thread_id] += data[thread_id+step];
                data2[thread_id] += data2[thread_id+step];
            }
            step *= 2;
            grab_id *= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (thread_id == 0) {
        
            // we now know how much needs to be reflected and adsorbed
            
            reflected_flux.x = data[0].x;
            adsorbed_flux.x  = data[0].y;
            
            // calculate the desorption rate based on the Langmuir isotherm
            double gamma_b = GAMMA_F*(1.0/VARTHETA_LANGMUIR - 1.0)*data2[0];
            
            desorbed_flux.x  = gamma_b*vartheta;
            
            double dvtheta = dt*ALPHA_P*(adsorbed_flux.x - desorbed_flux.x);
            
            //printf("dvtheta = %g\n",dvtheta);
            
            // make sure we don't get negative ratio of coverage
            // also ensure that we only desorb what is available, not 
            //  including what we have just adsorbed in this time step
            if ((vartheta + dvtheta) < 0.0) {
                desorbed_flux.x = vartheta/(dt*ALPHA_P);
                dvtheta = dt*ALPHA_P*(adsorbed_flux.x - desorbed_flux.x);
                
            }
            
            //printf("adsorbed = %g\ndesorbed = %g\nreflected = %g\n\n",adsorbed_flux.x,desorbed_flux.x,reflected_flux.x);

        
            COVER(face, ci).s0 += dvtheta;
            COVER(face, ci).s1 = reflected_flux.x;
            COVER(face, ci).s2 = adsorbed_flux.x;
            COVER(face, ci).s3 = desorbed_flux.x;
        }
    }
    
    return;
}

#endif

#if HAS_ACCOMMODATING_WALL == 2  // adsorb_CL

__kernel void
adsorbingWallCL_P2(__global double2* normal,
                    int face,
                    __global double4* wall_prop,
                    __global double4* wall_cover,
                    __global double2* wall_dist,
                    __global double2* flux_f,
                    __global double4* macro,
                    double dt)
{
    // Calculate the incoming velocity distribution that is to be 
    // reflected from the wall. 
    
    size_t gi = get_global_id(0);
    size_t gj = get_global_id(1);
    size_t thread_id = get_local_id(2);
    size_t ci;
    
    int rot;
    int face_id;

    switch (face) {
        case GNORTH:
            ci = gi;
            gi += GHOST;
            gj += NJ - GHOST;
            rot = -1;
            face_id = SOUTH;
            break;
        case GEAST:
            ci = gj;
            gi += NI - GHOST;
            gj += GHOST;
            rot = -1;
            face_id = WEST;
            break;
        case GSOUTH:
            ci = gi;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = SOUTH;
            break;
        case GWEST:
            ci = gj;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = WEST;
            break;
    }
    
    __local double2 data[LOCAL_SIZE];

    if (((face_id == SOUTH) && ((gi - GHOST) < ni)) 
    || ((face_id == WEST) && ((gj - GHOST) < nj))) {

        // calculate the updated wall coverage

        double2 face_normal = NORMAL(gi,gj,face_id);
        
        double4 wall = WALL_PROP(face, ci);
        wall.s12 = toLocal(wall.s12, face_normal);
        
        double reflected_flux = COVER(face, ci).s1;
        
        data[thread_id] = 0.0;
        
        // now calculate the Cercignani - Lampis distribution for the 
        // reflected flux
        double ratio = 1.0;
        if (reflected_flux > 0.0) {
            double2 CL_v, CL;
            for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
                size_t gv = li*LOCAL_SIZE+thread_id;
                if (gv < NV) {
                    double2 uv_out = interfaceVelocity(gv, face_normal); // velocities out of wall
                    int delta = (sign(uv_out.x)*rot + 1)/2; // (1 -> out of wall)
                    
                    // calculate the CL distribution for this velocity
                    CL_v = 0.0;
                    if (delta) {
                        for (size_t va = 0; va < NV; ++va) {
                            double2 uv_in = interfaceVelocity(va, face_normal); // velocities into the wall
                            int delta_in = (sign(uv_in.x)*rot + 1)/2; // (0 -> into wall)
                            
                            if ((!delta_in) && (uv_in.x != 0.0)) {
                                CL = CercignaniLampis(uv_in, uv_out, WALL_DIST(ci, va), ALPHA_N, ALPHA_T, 1.0/wall.s3);
                                CL.y += 0.5*ALPHA_T*(2-ALPHA_T)/wall.s3*CL.x;
                                CL_v += CL;
                            }
                        }
                        
                        CL_v *= WEIGHT[gv];
                    }
                    
                    double2 face_dist = FLUXF(gi,gj,gv);
                    FLUXF(gi,gj,gv) = delta*CL_v + (1-delta)*face_dist;
                    
                    data[thread_id] += rot*uv_out.x*CL_v;
                }
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            

            // perform reduction
            int step = 1;
            int grab_id = 2;
            while (step < LOCAL_SIZE) {
                if (!(thread_id%grab_id)) { // assume LOCAL_SIZE is a power of two
                    // reduce
                    data[thread_id] += data[thread_id+step];
                }
                step *= 2;
                grab_id *= 2;
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            // now we know how much the CL flux out of the wall is and can 
            // compare it to 'reflected_flux'
            
            if (thread_id == 0) {
                data[0].x = reflected_flux/data[0].x;
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            ratio = data[0].x;
        }
        
        COVER(face, ci).s2 = ratio;

        // now adjust the Cercignani - Lampis distribution so that we conserve mass
        // also add on any desorbing flux
        // convert from flux to density of distribution assuming that we 
        // have a Maxwellian (this is true for the desorbed particles)
        double desorbed_flux = COVER(face, ci).s3;
        wall.s0 = 2*desorbed_flux*sqrt(PI*wall.s3);
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                double2 uv = interfaceVelocity(gv, face_normal);
                int delta = (sign(uv.x)*rot + 1)/2; // (1 -> out of wall)
                
                double2 face_dist = FLUXF(gi,gj,gv);
                
                face_dist *= ((1-delta) + ratio*delta);
                 
                face_dist += delta*fM(wall, uv, gv);
                 
                FLUXF(gi,gj,gv) = face_dist;
            }
        } 
    }
    
    return;
}

#endif

#if HAS_ACCOMMODATING_WALL == 3  // adsorb_specular-diffuse
__kernel void
adsorbingWallDS_P2(__global double2* normal,
                    int face,
                    __global double4* wall_prop,
                    __global double4* wall_cover,
                    __global double2* wall_dist,
                    __global double2* flux_f,
                    __global double4* macro,
                    double dt)
{
    // Calculate the incoming velocity distribution that is to be 
    // reflected from the wall. 
    
    size_t gi = get_global_id(0);
    size_t gj = get_global_id(1);
    size_t thread_id = get_local_id(2);
    size_t ci;
    
    int rot;
    int face_id;

    switch (face) {
        case GNORTH:
            ci = gi;
            gi += GHOST;
            gj += NJ - GHOST;
            rot = -1;
            face_id = SOUTH;
            break;
        case GEAST:
            ci = gj;
            gi += NI - GHOST;
            gj += GHOST;
            rot = -1;
            face_id = WEST;
            break;
        case GSOUTH:
            ci = gi;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = SOUTH;
            break;
        case GWEST:
            ci = gj;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = WEST;
            break;
    }
    
    __local double2 data[LOCAL_SIZE];

    if (((face_id == SOUTH) && ((gi - GHOST) < ni)) 
    || ((face_id == WEST) && ((gj - GHOST) < nj))) {

        // read the updated wall coverage

        double2 face_normal = NORMAL(gi,gj,face_id);
        
        double4 wall = WALL_PROP(face, ci);
        wall.s12 = toLocal(wall.s12, face_normal);
        
        double reflected_flux = COVER(face, ci).s1;
        
        data[thread_id] = 0.0;
        
        // now calculate the specularly reflected distribution
        // NOTE: this only works for axis aligned boundaries
        double ratio = 1.0;
        if (reflected_flux > 0.0) {
            double2 specular;
            for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
                size_t gv = li*LOCAL_SIZE+thread_id;
                if (gv < NV) {
                    double2 uv_out = interfaceVelocity(gv, face_normal); // velocities out of wall
                    int delta = (sign(uv_out.x)*rot + 1)/2; // (1 -> out of wall)
                    
                    // calculate the distribution for this velocity
                    specular = 0.0;
                    int mv;
                    if (delta) {
                        if (face_id == WEST) {
                            mv = mirror_EW[gv];
                        } else {
                            mv = mirror_NS[gv];
                        }
                        specular = WALL_DIST(ci, mv);
                    }
                    
                    
                    
                    double2 face_dist = FLUXF(gi,gj,gv);
                    FLUXF(gi,gj,gv) = delta*specular + (1-delta)*face_dist;
                    
                    data[thread_id] += rot*uv_out.x*specular;
                }
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            

            // perform reduction
            int step = 1;
            int grab_id = 2;
            while (step < LOCAL_SIZE) {
                if (!(thread_id%grab_id)) { // assume LOCAL_SIZE is a power of two
                    // reduce
                    data[thread_id] += data[thread_id+step];
                }
                step *= 2;
                grab_id *= 2;
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            // now we know how much the specular flux out of the wall is and can 
            // compare it to 'reflected_flux'
            
            if (thread_id == 0) {
                data[0].x = reflected_flux/data[0].x;
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            ratio = data[0].x;
        }
        
        //COVER(face, ci).s2 = ratio;

        // now adjust the reflected distribution so that we conserve mass
        // also add on any desorbing flux
        
        // convert from flux to density of distribution assuming that we 
        // have a Maxwellian (this is true for the desorbed particles)
        double desorbed_flux = COVER(face, ci).s3;
        double adsorbed_flux = COVER(face, ci).s2;
        
        size_t account_adsorb = 0;
        if (((desorbed_flux == 0.0) && (adsorbed_flux > 0.0)) && (reflected_flux == 0.0)) {
            account_adsorb = 1;
        }
        
        wall.s0 = 2*desorbed_flux*sqrt(PI*wall.s3);
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                double2 uv = interfaceVelocity(gv, face_normal);
                int delta = (sign(uv.x)*rot + 1)/2; // (1 -> out of wall, 0 -> into wall)
                double2 face_dist = FLUXF(gi,gj,gv);
                face_dist *= 1 - delta*(1 - ratio);
                face_dist += delta*fM(wall, uv, gv);
                face_dist -= account_adsorb*delta*face_dist;
                FLUXF(gi,gj,gv) = face_dist;
            }
        }
        
        
        
        
    }
    
    return;
}

#endif

#if HAS_ACCOMMODATING_WALL > 0
__kernel void
wallFlux(__global double2* normal,
            __global double* side_length, int face,
            __global double2* flux_f, __global double4* flux_macro, 
            double dt)
{
    // given the flux information for each wall of each cell
    // modify the fluxes at the defined wall to give a diffuse wall
    

    size_t gi = get_global_id(0);
    size_t gj = get_global_id(1);
    size_t thread_id = get_local_id(2);
    
    int face_id;

    switch (face) {
        case GNORTH:
            gi += GHOST;
            gj += NJ - GHOST;
            face_id = SOUTH;
            break;
        case GEAST:
            gi += NI - GHOST;
            gj += GHOST;
            face_id = WEST;
            break;
        case GSOUTH:
            gi += GHOST;
            gj += GHOST;
            face_id = SOUTH;
            break;
        case GWEST:
            gi += GHOST;
            gj += GHOST;
            face_id = WEST;
            break;
    }
    
    __local double4 data[LOCAL_SIZE];

    if (((face_id == SOUTH) && ((gi - GHOST) < ni)) 
    || ((face_id == WEST) && ((gj - GHOST) < nj))) {

        // get the interface distribution and the flux out due to this distribution

        double2 face_normal = NORMAL(gi,gj,face_id);

        double face_length = LENGTH(gi,gj,face_id);

        // calculate the flux that is to come back into the domain given the distribution on the wall
        data[thread_id] = 0.0;
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                double2 uv = interfaceVelocity(gv, face_normal);
                
                double2 wall_dist = FLUXF(gi,gj,gv);
                
                FLUXF(gi,gj,gv) = uv.x*wall_dist*face_length*dt;
                
                data[thread_id].s0 += uv.x*wall_dist.x;
                data[thread_id].s1 += uv.x*uv.x*wall_dist.x;
                data[thread_id].s2 += uv.x*uv.y*wall_dist.x;
                data[thread_id].s3 += 0.5*uv.x*(dot(uv,uv)*wall_dist.x + wall_dist.y);
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // perform reduction
        
        int step = 1;
        int grab_id = 2;
        while (step < LOCAL_SIZE) {
            if (!(thread_id%grab_id)) { // assume LOCAL_SIZE is a power of two
                // reduce
                data[thread_id] += data[thread_id+step];
            }
            step *= 2;
            grab_id *= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (thread_id == 0) {
            // convert macro to global frame
            double4 macro_flux = data[0];
            
            macro_flux.s12 = toGlobal(macro_flux.s12, face_normal);

            macro_flux *= dt*face_length;

            FLUXM(gi,gj) = macro_flux;
        }
    }

    return;

}
#endif




#define RES(i,j) residual[(i)*nj + (j)]

#define FLUXFS(i,j,v) flux_f_S[NV*NJ*(i) + NV*(j) + (v)]
#define FLUXFW(i,j,v) flux_f_W[NV*NJ*(i) + NV*(j) + (v)]
#define FLUXMS(i,j) flux_macro_S[(i)*NJ + (j)]
#define FLUXMW(i,j) flux_macro_W[(i)*NJ + (j)]

__kernel void
updateMacro(__global double4* flux_macro_S, 
            __global double4* flux_macro_W,
            __global double* area,
            __global double4* macro,
            __global double4* residual)
{
    // update the macro buffer
    
    size_t mi = get_global_id(0);
    size_t mj = get_global_id(1);

    if ((mi < ni) && (mj < nj)) {
  
        int gi = mi + GHOST;
        int gj = mj + GHOST;
        
        double4 prim_old = MACRO(mi,mj);
        double4 w_old = getConserved(prim_old);
        
        double A = AREA(gi,gj);
        double4 w = w_old + (FLUXMS(gi, gj) - FLUXMS(gi, gj+1) + FLUXMW(gi,gj) - FLUXMW(gi+1,gj))/A;
        
        MACRO(mi,mj) = getPrimary(w);
        
        RES(mi,mj) = prim_old;  // use residual buffer to store the old data
    }
    
    return;
}

__kernel void
UGKS_update(__global double2* Fin,
	   __global double2* flux_f_S, __global double2* flux_f_W,
       __global double* area,
       __global double4* macro,
       __global double2* gQ,
       __global double4* residual,
       double dt,
       __global int* nan_check)
{
    // update the macro-properties and distribution
    
    size_t mi, mj, gi, gj, gv, thread_id;
    
    mi = get_global_id(0);
    mj = get_global_id(1);
    thread_id = get_local_id(2);
    
    gi = mi + GHOST;
    gj = mj + GHOST;
    
    // old stuff at t_n
    double4 prim_old = RES(mi,mj);
    double tau_old =  relaxTime(prim_old);
    double2 Q = GQ(mi, mj);
    
    // new relaxation rate
    double4 prim = MACRO(mi,mj);
    double tau = relaxTime(prim);
    
    double A = AREA(gi, gj);
    
    
    for (size_t loop_id = 0; loop_id < LOCAL_LOOP_LENGTH; ++loop_id) {

        gv = loop_id*LOCAL_SIZE + thread_id;

        if (gv < NV) {
        
            // old equilibrium
            double2 feq_old = fEQ(prim_old, Q, QUAD[gv], gv);
            
            // new stuff at t^n+1
            double2 feq = fEQ(prim, Q, QUAD[gv], gv);
            
            // update the distribution function
            
            double2 f_old = F(gi,gj,gv);
            double2 f_new = (f_old + (FLUXFS(gi,gj,gv) - FLUXFS(gi,gj+1,gv) + FLUXFW(gi,gj,gv) - FLUXFW(gi+1,gj,gv))/A
                          + 0.5*dt*(feq/tau+(feq_old-f_old)/tau_old))/(1.0+0.5*dt/tau);
                          
            if (any(isnan(f_new))) {
                nan_check[0] = 1;
            }
            
            F(gi,gj,gv) = f_new;
        }
        
    }
    
    return;
}

__kernel void
getResidual(__global double4* macro,
            __global double4* residual)
{
    // update the macro buffer
    
    size_t mi = get_global_id(0);
    size_t mj = get_global_id(1);

    if ((mi < ni) && (mj < nj)) {
        
        double4 prim_new = MACRO(mi,mj);
        double4 prim_old = RES(mi,mj);
        
        RES(mi,mj) = (prim_new - prim_old)*(prim_new - prim_old);
    }
    
    return;
}
