


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
  double N0, N1, N2, N3, N4, N5, N6;
  
  // pre-computed common terms
  
  double dd = dot(uv,uv);
  double dd2 = dd*dd;
  double dd3 = dd*dd2;
  
  double DD = dot(UV,UV);
  double DD2 = DD*DD;
  double DD3 = DD2*DD;
  
  double dD = dot(uv,UV);
  double dD2 = dD*dD;
  double dD4 = dD2*dD2;
  double dD6 = dD2*dD4;
  
  double T2 = T*T;
  double T3 = T2*T;
  
  double T1 = T - 1.0;
  double T12 = T1*T1;
  double T13 = T12*T1;
  
  //equilibrium distribution function
  // zero-th order
  N0 = 1.0;
  
  // first order
  N1 = dD;
  
  // second order
  N2 = (1./2.)*(dD2 - DD + (T-1.0)*(dd-2.0));
  
  // third order
  N3 = (1./6.)*dD*(3*(T-1.0)*(dd-4.0)+dD2-3*DD);
  
  // fourth order
  N4 = (3*DD2 + dD4 + 6*(-6 + dd)*dD2*T1 - 6*DD*(dD2 + (-4 + dd)*T1) + 3*(8 + (-8 + dd)*dd)*T12)/24.;
  
  // fifth order
  N5 = (dD*(15*DD2 + dD4 + 10*(-8 + dd)*dD2*T1 - 10*DD*(dD2 + 3*(-6 + dd)*T1) + 15*(24 + (-12 + dd)*dd)*T12))/120.;
	
  // sixth order
  N6 = (720 - 1080*DD + 45*(-12 + DD)*(-4 + DD)*dD2 + 270*DD2 - 15*DD3 - 15*(-10 + DD)*dD4 + dD6 - 30*(9*(8 + 
      (-8 + DD)*DD) - 24*(-6 + DD)*dD2 + 5*dD4)*T - 45*dd2*(-6 + DD - dD2 + 6*T)*T12 + 15*dd*T1*(3*DD2 + dD4 - 
      6*DD*(6 + dD2 - 6*T) - 48*dD2*T1 + 72*T12) + 15*dd3*T13 - 1080*(DD - 2*(1 + dD2))*T2 - 720*T3)/720.;
  
  double2 out;

  out.x = rho*(N0 + N1 + N2 + N3 + N4 + N5 + N6);
    
  out.y = K*T*out.x;
  
  return ghW[i]*out;
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
  S.x = (1.0 - ((Pr - 1.0)*(qx*(u - U) 
  + qy*(v - V))*((u - U)*(u - U) + (v - V)*(v - V) - 4.0*T))
  /(5.0*T*T*T*rho));
  
  S.y = (1.0 - ((Pr - 1.0)*(qx*(u - U) 
  + qy*(v - V))*((u - U)*(u - U) + (v - V)*(v - V) - 2.0*T))
  /(5.0*T*T*T*rho));
  
  return S;
}

double2 fS(double rho, double2 UV, double T, double2 Q, double2 uv, size_t i)
{
  //Shakhov equation for f
  double2 out = fM(rho, UV, T, uv, i)*fShakhov(rho, UV, T, Q, uv);
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// regNEq
////////////////////////////////////////////////////////////////////////////////

double2 regNEq(double2 uv, double2* M, size_t i)
{
  // regularized non-equilibrium distribution
  
  double u = uv.x; 
  double v = uv.y;
  
  double2 N0, N1, N2, N3, N4, N5, N6;
  double u2, v2, u3, v3, u4, v4, u5, v5, u6, v6;
  double weight;
  double2 out;
  
  // pre-computed velocity combinations
  u2 = u*u;
  u3 = u2*u;
  u4 = u3*u;
  u5 = u4*u;
  u6 = u5*u;
  
  v2 = v*v;
  v3 = v2*v;
  v4 = v3*v;
  v5 = v4*v;
  v6 = v5*v;
  
  // zero-th order
  N0 = M[0];
  
  // first order
  N1 = M[1]*u + M[2]*v;
  
  // second order
  N2 = ((-M[0] + M[4])*(-1.0 + u2) + 2.0*M[3]*u*v + (-M[0] + M[5])*(-1.0 + v2))/2.0;
  
  // third order
  N3 = ((-3*M[1] + M[8])*u*(-3 + u2) + 3*(-M[2] + M[7])*(-1 + u2)*v + 
	(-3*M[2] + M[9])*v*(-3 + v2) + 3*(-M[1] + M[6])*u*(-1 + v2))/6.;
  
  // fourth order
  N4 = ((3*M[0] + M[13] - 6*M[4])*(3 - 6*u2 + u4) + 6*(M[0] + M[12] - M[4]
	- M[5])*(-1 + u2)*(-1 + v2) + (3*M[0] + M[14] - 6*M[5])*(3 - 
	6*v2 + v4) + 4*u*v*((M[10] - 3*M[3])*(-3 + u2) + (M[11] - 3*M[3])*(-3 + v2)))/24.;
  
  // fifth order
  N5 = ((15*M[1] + M[19] - 10*M[8])*u*(15 - 10*u2 + u4) + 5*(M[17] 
	+ 3*M[2] - 6*M[7])*(3 - 6*u2 + u4)*v + 10*(M[16] + 3*M[2] - 
	3*M[7] - M[9])*(-1 + u2)*v*(-3 + v2) + 10*(3*M[1] + M[15] - 
	3*M[6] - M[8])*u*(-3 + u2)*(-1 + v2) + (15*M[2] + M[20] - 
	10*M[9])*v*(15 - 10*v2 + v4) + 5*(3*M[1] + M[18] - 6*M[6])*u*(3 - 
	6*v2 + v4))/120.;
	
  // sixth order
  N6 = ((-15*M[0] - 15*M[13] + M[26] + 45*M[4])*(-15 + 45*u2 - 15*u4 
	+ u6) + 6*(-10*M[10] + M[25] + 15*M[3])*u*(15 - 10*u2 + u4)*v 
	+ 20*(-3*M[10] - 3*M[11] + M[23] + 9*M[3])*u*(-3 + u2)*v*(-3 + v2) + 
	15*(-3*M[0] - 6*M[12] - M[13] + M[22] + 6*M[4] + 3*M[5])*(3 - 6*u2 
	+ u4)*(-1 + v2) + 6*(-10*M[11] + M[24] + 15*M[3])*u*v*(15 - 10*v2 + 
	v4) + 15*(-3*M[0] - 6*M[12] - M[14] + M[21] + 3*M[4] + 6*M[5])*(-1 + u2)*(3
	- 6*v2 + v4) + (-15*M[0] - 15*M[14] + M[27] + 45*M[5])*(-15 + 45*v2
	- 15*v4 + v6))/720.;
	
  weight = ghW[i];
  
  out = weight*(N0 + N1 + N2 + N3 + N4 + N5 + N6);
  
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// getMoments
////////////////////////////////////////////////////////////////////////////////

int getMoments(__global double2* Fin, size_t i, size_t j, double2* M, double* rho, double2* UV, double* T, double2* Q)
{
  // compute all required moments of the distribution
  
  // initialise M to zero
  for (size_t ii = 0; ii < 28; ii++) {
    M[ii] = 0.0;
  }
  
  double2 f;
  double u, v, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6; //NOTE: all of these could be pre-computed, would take a bit of space though
  double2 uf, vf, u2f, v2f;
  
  for (size_t ii = 0; ii < NV; ii++) {
    
    // velocities
    u = ghV[ii].x;
    v = ghV[ii].y;
    
    // distribution
    f = F(i,j,ii);
    
    // pre-computed velocity combinations
    u2 = u*u;
    u3 = u2*u;
    u4 = u3*u;
    u5 = u4*u;
    u6 = u5*u;
    
    v2 = v*v;
    v3 = v2*v;
    v4 = v3*v;
    v5 = v4*v;
    v6 = v5*v;
    
    uf = u*f;
    vf = v*f;
    
    u2f = u2*f;
    v2f = v2*f;
    
    // moments
    M[0] += f;
    M[1] += uf;
    M[2] += vf;
    M[3] += u*vf;
    M[4] += u2f;
    M[5] += v2f;
    M[6] += v2*uf;
    M[7] += u2*vf;
    M[8] += u3*f;
    M[9] += v3*f;
    M[10] += u3*vf;
    M[11] += v3*uf;
    M[12] += u2*v2f;
    M[13] += u4*f;
    M[14] += v4*f;
    M[15] += u3*v2f;
    M[16] += v3*u2f;
    M[17] += u4*vf;
    M[18] += v4*uf;
    M[19] += u5*f;
    M[20] += v5*f;
    M[21] += v4*u2f;
    M[22] += u4*v2f;
    M[23] += u3*v3*f;
    M[24] += v5*uf;
    M[25] += u5*vf;
    M[26] += u6*f;
    M[27] += v6*f;
  }
  
  double D, U, U2, V, V2, E, Txyz, nrg, qx, qy;
  
  D = M[0].x;   //density
  U = M[1].x/D;	// mean velocity x
  V = M[2].x/D;	// mean velocity y
  U2 = U*U;
  V2 = V*V;
  E = (M[4].x + M[5].x + M[0].y)/D;   // 2 x energy
  Txyz = (E - U2 - V2)/B;	// temperature
  nrg = U2 + V2 - 3.0*Txyz;
  qx = 0.5*((M[8].x + M[6].x + M[1].y/K) + D*U*nrg) - U*M[4].x - V*M[3].x;  //heat flux -x
  qy = 0.5*((M[7].x + M[9].x + M[2].y/K) + D*V*nrg) - U*M[3].x - V*M[5].x;  //heat flux -y
  
  (*rho) = D;
  (*UV).x = U;
  (*UV).y = V;
  (*T) = Txyz;
  (*Q).x = qx;
  (*Q).y = qy;
  
  return 0;
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
    double u = ghV[ii].x;
    double v = ghV[ii].y;
    
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
    u = ghV[ii].x;
    v = ghV[ii].y;
    
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
    double u = ghV[ii].x;
    double v = ghV[ii].y;
    
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
  double2 e = ghV[v];
  
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

