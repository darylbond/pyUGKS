


/////////////////////////////////////////
// relaxTime
/////////////////////////////////////////

double relaxTime(double rho, double T)
{
  // calculate the non-dimensional relaxation time
  
  double tau = (Kn/rho)*sqrt(2.0/PI)*pow(T,chi - 1.0);
  //double tau = (5./8.)*(Kn/rho)*sqrt(2.0/PI)*pow(T,chi - 1.0);
  
  return tau;
}



/////////////////////////////////////////
// MAXWELL-BOLTZMANN EQUILIBRIUM
/////////////////////////////////////////
    
double2 fM(double rho, double2 UV, double T, double2 uv, size_t i) 
{  
  // the Maxwellian
  
  double2 M;

  M.x = (rho/(T*PI))*exp(-dot(uv-UV,uv-UV)/T);
  
  M.y = (M.x*K*T)/2.0;
  
  return WEIGHT[i]*M;
}
    
////////////////////////////////////////////////////////////////////////////////
// SHAKHOV EQUATIONS
////////////////////////////////////////////////////////////////////////////////

double2 fShakhov(double rho, double2 UV, double T, double2 Q, double2 uv)
{
  // Shakhov extension
  double U = UV.x; 
  double V = UV.y; 
  double qx = Q.x; 
  double qy = Q.y;
  
  double u = uv.x; 
  double v = uv.y;
  
  double2 S;
  S.x = (0.8*(1 - Pr)*(K - 5 + (2*((u - U)*(u - U) + (v - V)*(v - V)))/T)*
        (qx*(u - U) + qy*(v - V)))/(rho*T*T);
  
  S.y = (0.8*(1 - Pr)*(K - 3 + (2*((u - U)*(u - U) + (v - V)*(v - V)))/T)*
        (qx*(u - U) + qy*(v - V)))/(rho*T*T);
  
  return S;
}

double2 fS(double rho, double2 UV, double T, double2 Q, double2 uv, size_t i)
{
  //Shakhov equation for f
  double2 out = fM(rho, UV, T, uv, i)*(1 + fShakhov(rho, UV, T, Q, uv));
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// macroProp
////////////////////////////////////////////////////////////////////////////////

int macroProp(__global double2* Fin, size_t i, size_t j, double* rho, double2* UV, double* T, double2* Q)
{
  // return the conserved properties for the index (i,j)
  
  double2 M0, M1, M2;
  double M3, M4, M5, M6, M7, M8, M9;  
  
  M0 = 0.0;  M1 = 0.0;  M2 = 0.0;  M3 = 0.0;  M4 = 0.0;
  M5 = 0.0;  M6 = 0.0;  M7 = 0.0;  M8 = 0.0;  M9 = 0.0;
  
  for (size_t ii = 0; ii < NV; ii++) {
    
    // velocities
    double u = QUAD[ii].x;
    double v = QUAD[ii].y;
    
    // distribution
    double2 f = F(i,j,ii);
    
    // moments
    M0 += f;
    M1 += u*f;
    M2 += v*f;
    M3 += u*v*f.x;
    M4 += u*u*f.x;
    M5 += v*v*f.x;
    M6 += v*v*u*f.x;
    M7 += u*u*v*f.x;
    M8 += u*u*u*f.x;
    M9 += v*v*v*f.x;
  }
  
  (*rho) = M0.x;
  (*UV).x = M1.x/M0.x;
  (*UV).y = M2.x/M0.x;
  (*T) = ((M4 + M5 + M0.y)/M0.x - (*UV).x*(*UV).x - (*UV).y*(*UV).y)/B;	// temperature
  double nrg = (*UV).x*(*UV).x + (*UV).y*(*UV).y - 3.0*(*T);
  (*Q).x = 0.5*((M8 + M6 + M1.y/K) + M0.x*(*UV).x*nrg) - (*UV).x*M4 - (*UV).y*M3;  //heat flux -x
  (*Q).y = 0.5*((M7 + M9 + M2.y/K) + M0.x*(*UV).y*nrg) - (*UV).x*M3 - (*UV).y*M5;  //heat flux -y
  
  /*
  double D, U, U2, V, V2, E, Txyz, nrg, qx, qy;
  D = M0.x;   //density
  U = M1.x/D;	// mean velocity x
  V = M2.x/D;	// mean velocity y
  U2 = U*U;
  V2 = V*V;
  E = (M4.x + M5.x + M0.y)/D;   // 2 x energy
  Txyz = (E - U2 - V2)/B;	// temperature
  nrg = U2 + V2 - 3.0*Txyz;  
  qx = 0.5*((M8.x + M6.x + M1.y/K) + D*U*nrg) - U*M4.x - V*M3.x;  //heat flux -x
  qy = 0.5*((M7.x + M9.x + M2.y/K) + D*V*nrg) - U*M3.x - V*M5.x;  //heat flux -y
  */
  return 0;
}

int macroPropLocal(__local double2 fdist[WSI][WSJ][NV], double* rho, double2* UV, double* T, double2* Q)
{
  // return the conserved properties for the index (i,j)
  
  int ti = get_local_id(0);
  int tj = get_local_id(1);
  
  double2 f;
  double2 M0, M1, M2;
  double M3, M4, M5, M6, M7, M8, M9;  
  double u, v;
  
  M0 = 0.0;  M1 = 0.0;  M2 = 0.0;  M3 = 0.0;  M4 = 0.0;
  M5 = 0.0;  M6 = 0.0;  M7 = 0.0;  M8 = 0.0;  M9 = 0.0;
  
  for (size_t ii = 0; ii < NV; ii++) {
    
    // velocities
    u = QUAD[ii].x;
    v = QUAD[ii].y;
    
    // distribution
    f = fdist[ti][tj][ii];
    
    // moments
    M0 += f;
    M1 += u*f;
    M2 += v*f;
    M3 += u*v*f.x;
    M4 += u*u*f.x;
    M5 += v*v*f.x;
    M6 += v*v*u*f.x;
    M7 += u*u*v*f.x;
    M8 += u*u*u*f.x;
    M9 += v*v*v*f.x;
  }
  
  (*rho) = M0.x;
  (*UV).x = M1.x/M0.x;
  (*UV).y = M2.x/M0.x;
  (*T) = ((M4 + M5 + M0.y)/M0.x - (*UV).x*(*UV).x - (*UV).y*(*UV).y)/B;	// temperature
  double nrg = (*UV).x*(*UV).x + (*UV).y*(*UV).y - 3.0*(*T);
  (*Q).x = 0.5*((M8 + M6 + M1.y/K) + M0.x*(*UV).x*nrg) - (*UV).x*M4 - (*UV).y*M3;  //heat flux -x
  (*Q).y = 0.5*((M7 + M9 + M2.y/K) + M0.x*(*UV).y*nrg) - (*UV).x*M3 - (*UV).y*M5;  //heat flux -y
  
  return 0;
}

int macroShort(__global double2* Fin, size_t i, size_t j, double* rho, double2* UV, double* T)
{
  // return the conserved properties for the index (i,j)
  
  double2 M0, M1, M2;
  double M3, M4, M5;  
  
  M0 = 0.0;  M1 = 0.0;  M2 = 0.0;  M3 = 0.0;  M4 = 0.0;
  M5 = 0.0;
  
  for (size_t ii = 0; ii < NV; ii++) {
    
    // velocities
    double u = QUAD[ii].x;
    double v = QUAD[ii].y;
    
    // distribution
    double2 f = F(i,j,ii);
    
    // moments
    M0 += f;
    M1 += u*f;
    M2 += v*f;
    M3 += u*v*f.x;
    M4 += u*u*f.x;
    M5 += v*v*f.x;
  }
  
  (*rho) = M0.x;
  (*UV).x = M1.x/M0.x;
  (*UV).y = M2.x/M0.x;
  (*T) = ((M4 + M5 + M0.y)/M0.x - (*UV).x*(*UV).x - (*UV).y*(*UV).y)/B;	// temperature
  return 0;
}

/////////////////////////////////////////
// interfaceVelocity
/////////////////////////////////////////

double2 interfaceVelocity(int i, int j, int v, int face, __global double2* normal)
{
  // find velocities aligned with the nominated face
  // u -> normal pointing out
  
  // global velocity
  double2 e = QUAD[v];
  
  // normal to edge
  double2 n = NORMAL(i,j,face);
  
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


/////////////////////////////////////////
// diffuseReflect
/////////////////////////////////////////
#if HAS_DIFFUSE_WALL == 1
void diffuseReflect(__local double2 cell_flux[WSI][WSJ][NV], 
                     __local double2 wall_flux[WSI][WSJ][NV],
                     __local double ratio[WSI][WSJ][1],
                     __local double wall_velocity[WSI][WSJ][NV],
                     int face, double2 uv, int flux_direction,
                     int gi, int gj)
{
  // update the values of cell_flux to follow the principles of diffuse reflection
  

  int ti = get_local_id(0);
  int tj = get_local_id(1);
  int gv = get_global_id(2);
  
  // wall macro-props:
  double Twall = BC_cond[face].s0;
  double2 UVwall;
  UVwall.x = 0.0; // wall velocity, normal to wall
  UVwall.y = BC_cond[face].s1; // wall velocity tangential
  
  // load uv into wall_velocity
  wall_velocity[ti][tj][gv] = uv.x;
  
  // the wall_flux is what is coming out of the wall into the interior domain
  double2 f0 = fM(1.0, UVwall, Twall, uv, gv);
  //NOTE: flux_direction = -1 when doing SOUTH and WEST
  //NOTE: flux_direction = +1 when doing NORTH and EAST
  wall_flux[ti][tj][gv] = fabs(0.5*(uv.x - flux_direction*fabs(uv.x)))*f0;
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // get the ratio of out to in
  double u;
  if (gv == 0) {
    double sumOUT = 0.0;
    double sumIN = 0.0;
    for (int v = 0; v < NV; v++) {
      u = sign(-flux_direction*wall_velocity[ti][tj][v]);
      sumOUT -= fabs((u - 1)/2.0)*cell_flux[ti][tj][v].x;
      sumIN += wall_flux[ti][tj][v].x;
    }

    // get the ratio of out / in
    ratio[ti][tj][0] = sumOUT / sumIN;
    
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // now feed "ratio * wall_flux" to "cell_flux"
  // but only the incoming fluxes
  u = sign(uv.x);
                                     // if flux into interior
  cell_flux[ti][tj][gv] = fabs((u - flux_direction)/2.0)*ratio[ti][tj][0]*wall_flux[ti][tj][gv]
                                        // if flux into wall
                          + fabs((u + flux_direction)/2.0)*cell_flux[ti][tj][gv];

  return;
}
#endif

