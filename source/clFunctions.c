


/////////////////////////////////////////
// relaxTime
/////////////////////////////////////////

double relaxTime(double4 prim)
{
  // calculate the non-dimensional relaxation time
  //double alpha_ref = 1.0;
  //double omega_ref = 0.5;
  //double mu_ref = 5*(alpha_ref+1)*(alpha_ref+2)*sqrt(PI)/(4*alpha_ref*(5-2*omega_ref)*(7-2*omega_ref))*Kn;
  //double tau = mu_ref*2*pow(prim.s3,1-chi)/prim.s0;
  
  double tau = (Kn/prim.s0)*sqrt(2.0/PI)*pow(prim.s3,1.0 - chi);
  
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
  
  S = 0.8*(1-Pr)*(prim.s3*prim.s3)/prim.s0*((uv.x-prim.s1)*Q.x+(uv.y-prim.s2)*Q.y);
  
  double part = 2*prim.s3*(dot(uv-prim.s12,uv-prim.s12))+K;
  
  S.x *= part-5;
  S.y *= part-3;
  
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
// toLocal
/////////////////////////////////////////

double2 toLocal(double2 in, double2 n)
{
  // find velocities aligned with the nominated face
  
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


