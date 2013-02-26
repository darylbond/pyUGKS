


/////////////////////////////////////////
// relaxTime
/////////////////////////////////////////

double relaxTime(double4 prim)
{
  // calculate the non-dimensional relaxation time
  double alpha_ref = 1.0;
  double omega_ref = 0.5;
  double mu_ref = 5*(alpha_ref+1)*(alpha_ref+2)*sqrt(PI)/(4*alpha_ref*(5-2*omega_ref)*(7-2*omega_ref))*Kn;
  double tau = mu_ref*2*pow(prim.s3,1-chi)/prim.s0;
  
  //double tau = (Kn/prim.s0)*sqrt(2.0/PI)*pow(prim.s3,1.0 - chi);
  
  //double tau = (Kn/rho)*sqrt(2.0/PI)*pow(T,chi - 1.0);
  //double tau = (5./8.)*(Kn/rho)*sqrt(2.0/PI)*pow(T,chi - 1.0);
  
  return tau;
}

/////////////////////////////////////////
// soundSpeed
/////////////////////////////////////////

double soundSpeed(double lambda)
{
  // calculate the sound speed
  
  return sqrt(0.5*gam/lambda);
}

/////////////////////////////////////////
// MAXWELL-BOLTZMANN EQUILIBRIUM
/////////////////////////////////////////
    
double2 fM(double4 prim, double2 uv, size_t gv) 
{  
  // the Maxwellian
  
  double2 M;
  M.x = prim.s0*(prim.s3/PI)*exp(-prim.s3*dot(uv-prim.s12,uv-prim.s12));
  M.y = (M.x*K)/(2.0*prim.s3);
  
  return WEIGHT[gv]*M;
}
    
////////////////////////////////////////////////////////////////////////////////
// SHAKHOV EQUATIONS
////////////////////////////////////////////////////////////////////////////////

double2 fS(double4 prim, double2 Q, double2 uv, double2 M)
{
  // Shakhov extension
    
  double2 S;
  S.x = 0.8*(1-Pr)*(prim.s3*prim.s3)/prim.s0*((uv.x-prim.s1)*Q.x+(uv.y-prim.s2)*Q.y)*(2*prim.s3*(((uv.x-prim.s1)*(uv.x-prim.s1))+((uv.y-prim.s2)*(uv.y-prim.s2)))+K-5);
  S.y = 0.8*(1-Pr)*(prim.s3*prim.s3)/prim.s0*((uv.x-prim.s1)*Q.x+(uv.y-prim.s2)*Q.y)*(2*prim.s3*(((uv.x-prim.s1)*(uv.x-prim.s1))+((uv.y-prim.s2)*(uv.y-prim.s2)))+K-3);
  
  return S*M;
}

double2 fEQ(double4 prim, double2 Q, double2 uv, size_t gv)
{
  // the full Shakhov equilibrium
  
  double2 M = fM(prim, uv, gv);
  
  return M + fS(prim, Q, uv, M);
}

/////////////////////////////////////////
// interfaceVelocity
/////////////////////////////////////////

double2 interfaceVelocity(int v, double2 n)
{
  // find velocities aligned with the nominated face
  // u -> normal pointing out
  
  // global velocity
  double2 e = QUAD[v];
  
  // tangent to edge
  double2 t;
  t.x = -n.y;
  t.y = n.x;
  
  double2 uv;
  uv.x = dot(e,n);
  uv.y = dot(e,t);
  
  return uv;
}

/////////////////////////////////////////
// toLocal
/////////////////////////////////////////
double2 toLocal(double2 in, double2 n)
{
  // tangent to edge
  double2 t;
  t.x = -n.y;
  t.y = n.x;
  
  double2 out;
  out.x = dot(in,n);
  out.y = dot(in,t);
  
  return out;
}
/////////////////////////////////////////
// toGlobal
/////////////////////////////////////////

double2 toGlobal(double2 in, double2 n)
{
  // change from coord system based on 'n'
  // to the cartesian system
  
  // tangent to edge
  double2 t;
  t.x = -n.y;
  t.y = n.x;
  
  double2 out;
  out.x = dot(n*in.x,(double2)(1.0,0.0)) + dot(t*in.y,(double2)(1.0,0.0));
  out.y = dot(n*in.x,(double2)(0.0,1.0)) + dot(t*in.y,(double2)(0.0,1.0));
  
  return out;
}

/////////////////////////////////////////
// vanLeer
/////////////////////////////////////////
#if FLUX_METHOD == 0
double2 vanLeer(double2 s1, double2 s2)
{
  // perform van Leer limiter
  return (sign(s1) + sign(s2))*(fabs(s1)*fabs(s2))/(fabs(s1) + fabs(s2) + 1e-16);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// WENO5
////////////////////////////////////////////////////////////////////////////////
#if FLUX_METHOD == 1
#define EPSILON 1E-16
double2 WENO5(double2* S, double dx)
{
    // calculate the linear gradient of the cell for one direction

    double2 B0, B1, B2, alpha0, alpha1, alpha2;
    double2 omega0, omega1, omega2, f0, f1, f2, temp1, temp2;
    
    size_t J = 2;

    temp1 = S[J-2] - 2.0*S[J-1] + S[J];
    temp2 = S[J-2] - 4.0*S[J-1] + 3.0*S[J];
    B0 = (13.0/12.0)*temp1*temp1 + 0.25*temp2*temp2;
    temp1 = S[J-1] - 2.0*S[J] + S[J+1];
    temp2 = S[J-1] - S[J+1];
    B1 = (13.0/12.0)*temp1*temp1 + 0.25*temp2*temp2;
    temp1 = S[J] - 2.0*S[J+1] + S[J+2];
    temp2 = 3.0*S[J] - 4.0*S[J+1] + S[J+2];
    B2 = (13.0/12.0)*temp1*temp1 + 0.25*temp2*temp2;

    temp1 = EPSILON + B0;
    alpha0 = 0.1/(temp1*temp1);
    
    temp1 = EPSILON + B1;
    alpha1 = 0.6/(temp1*temp1);
    
    temp1 = EPSILON + B2;
    alpha2 = 0.3/(temp1*temp1);
    
    temp1 = alpha0 + alpha1 + alpha2;

    omega0 = alpha0/temp1;
    omega1 = alpha1/temp1;
    omega2 = alpha2/temp1;

    f0 = (1./3.)*S[J-2] - (7./6.)*S[J-1] + (11./6.)*S[J];
    f1 = -(1./6.)*S[J-1] + (5./6.)*S[J] + (1./3.)*S[J+1];
    f2 = (1./3.)*S[J] + (5./6.)*S[J+1] - (1./6.)*S[J+2];

    double2 fL = omega0*f0 + omega1*f1 + omega2*f2;
    
    // calculate slope of f in cell J, assume linear slope
    // slope from centre to interface - so dx is distance from centre to interface
    
    return (fL - S[J])/dx;
}
#endif

double4 getInterfaceSingle(__global double2* Fin, __global double2* centre, __global double2* mid_side, size_t gi, size_t gj, size_t gv, int face, double2 face_normal) 
{
  // calculate the interface distribution using a choice of limiter

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

  double4 out;

  out.s01 = f_stencil[MID_STENCIL] + sigma*interface_distance;
  out.s23 = sigma;

  return out;
}

void getInterfaceDist(__global double2* Fin, __global double2* centre, __global double2* mid_side, size_t gi, size_t gj, int face, double2 face_normal, 
    __local double2 face_dist[NV], __local double2 face_slope[NV]) {
    // calculate the interface distribution using a choice of limiter

    for (size_t gv = 0; gv < NV; ++gv) {

        double4 f_sigma = getInterfaceSingle(Fin, centre, mid_side, gi, gj, gv, face, face_normal);

        // the interface value of f
        face_dist[gv] = f_sigma.s01;
        face_slope[gv] = f_sigma.s23;
    }
    
    return;
}


double2 getWallDistribution (__global double2* Fin, __global double2* centre, __global double2* mid_side, size_t gi, size_t gj, int rot, int face, double2 face_normal, double4 wall,
    __local double2 face_dist[NV], __local double2 wall_dist[NV])
{
  // calculate the wall distribution for a diffuse wall
  
  double2 sums = 0.0;
  
  for (size_t gv = 0; gv < NV; ++gv) {

        double4 f_sigma = getInterfaceSingle(Fin, centre, mid_side, gi, gj, gv, face, face_normal);
        
        // the interface value of f
        face_dist[gv] = f_sigma.s01;
        
        double2 uv = interfaceVelocity(gv, face_normal);
        
        int delta = (sign(uv.x)*rot + 1)/2;
        
        sums.x += uv.x*(1-delta)*f_sigma.x;
        
        wall_dist[gv] = fM(wall, uv, gv);
        
        sums.y -= uv.x*delta*wall_dist[gv].x;
    }
  
  return sums;
}

