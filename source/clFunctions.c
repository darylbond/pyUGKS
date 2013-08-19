


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
  
  #if RELAX_TYPE == 0
  double tau = (Kn/prim.s0)*sqrt(2.0/PI)*pow(prim.s3,1.0 - chi);
  #endif
  
  #if RELAX_TYPE == 1
  double tau =  (5./8.)*(Kn/prim.s0)*sqrt(PI)*pow(prim.s3,1.0 - chi);
  #endif
  
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
  
  S = 0.8*(1-Pr)*(prim.s3*prim.s3)/prim.s0*dot(uv-prim.s12,Q);
  
  double part = 2*prim.s3*(dot(uv-prim.s12,uv-prim.s12));
  
  S.x *= part-4;
  S.y *= part-2;
  
  return S*M;
}

double2 fEQ(double4 prim, double2 Q, double2 uv, size_t gv)
{
  // the full Shakhov equilibrium
  
  double2 M = fM(prim, uv, gv);
  
  return M + fS(prim, Q, uv, M);
}

double bessi0(double x) { 
	// -- See paper J.M. Blair, "Rational Chebyshev approximations for the modified Bessel 
	// functions I_0(x) and I_1(x)", Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.
	double num, den, x2;
	x2 = fabs(x*x);    x=fabs(x);
  if ( x == 0.0) {
    return 1.0;
	} else if (x > 15.0) {
		den = 1.0 / x;
		num =              -4.4979236558557991E+006;
		num = fma (num, den,  2.7472555659426521E+006);
		num = fma (num, den, -6.4572046640793153E+005);
		num = fma (num, den,  8.5476214845610564E+004);
		num = fma (num, den, -7.1127665397362362E+003);
		num = fma (num, den,  4.1710918140001479E+002);
		num = fma (num, den, -1.3787683843558749E+001);
		num = fma (num, den,  1.1452802345029696E+000);
		num = fma (num, den,  2.1935487807470277E-001);
		num = fma (num, den,  9.0727240339987830E-002);
		num = fma (num, den,  4.4741066428061006E-002);
		num = fma (num, den,  2.9219412078729436E-002);
		num = fma (num, den,  2.8050629067165909E-002);
		num = fma (num, den,  4.9867785050221047E-002);  
		num = fma (num, den,  3.9894228040143265E-001);
		num = num * den;
		den = sqrt (x);
		num = num * den;
		den = exp (0.5 * x);  /* prevent premature overflow */
		num = num * den;
		num = num * den;
		return num;
	} else {  
		num = -0.28840544803647313855232E-028;
		num = fma (num, x2, -0.72585406935875957424755E-025);
		num = fma (num, x2, -0.1247819710175804058844059E-021);
		num = fma (num, x2, -0.15795544211478823152992269E-018);
		num = fma (num, x2, -0.15587387207852991014838679E-015); 
		num = fma (num, x2, -0.121992831543841162565677055E-012); 
		num = fma (num, x2, -0.760147559624348256501094832E-010);  
		num = fma (num, x2, -0.375114023744978945259642850E-007); 
		num = fma (num, x2, -0.1447896113298369009581404138E-004);
		num = fma (num, x2, -0.4287350374762007105516581810E-002);
		num = fma (num, x2, -0.947449149975326604416967031E+000); 
		num = fma (num, x2, -0.1503841142335444405893518061E+003); 
		num = fma (num, x2, -0.1624100026427837007503320319E+005); 
		num = fma (num, x2, -0.11016595146164611763171787004E+007);
		num = fma (num, x2, -0.4130296432630476829274339869E+008); 
		num = fma (num, x2, -0.6768549084673824894340380223E+009);  
		num = fma (num, x2, -0.27288446572737951578789523409E+010);
		den = 0.1E+001;
		den = fma (den, x2, -0.38305191682802536272760E+004); 
		den = fma (den, x2, 0.5356255851066290475987259E+007);  
		den = fma (den, x2, -0.2728844657273795156746641315E+010);
		return num/den; 
	} 
}

double scatterNormal(double c_in, double c_out, double alpha_n, double Tw)
{
  // normal scattering
  
  double Rn = (2.0/(alpha_n*Tw))*fabs(c_out)
        *bessi0(((2*sqrt(1-alpha_n))/(alpha_n*Tw))*c_in*c_out)
        *exp(-(c_out*c_out + (1-alpha_n)*c_in*c_in)/(alpha_n*Tw));
  
  return Rn;
}

double scatterTangential(double c_in, double c_out, double alpha_t, double Tw)
{
  // tangential scattering
  
  double a = c_out - (1-alpha_t)*c_in;
  
  double Rt = (1.0/sqrt(PI*alpha_t*(2-alpha_t)*Tw))
               *exp(-(a*a)/(alpha_t*(2-alpha_t)*Tw));
  
  return Rt;
}

double2 CercignaniLampis(double2 uv_in, double2 uv_out, double2 f_in, double alpha_n, double alpha_t, double Tw)
{
  // calculate the contribution of uv_in to uv_out
  
  double2 f_out;
  
  f_out = fabs(uv_in.x/uv_out.x)
             *scatterNormal(uv_in.x, uv_out.x, alpha_n, Tw)
             *scatterTangential(uv_in.y, uv_out.y, alpha_t, Tw);
             
             
  f_out *= f_in;
  
  f_out.y *= (1-alpha_t)*(1-alpha_t);
  
  return f_out;
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
  // find velocities aligned with the nominated normal
  
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

double2 transform(double2 a0, double2 a1, double2 b0, double2 b1, double2 n)
{
  // given [a,b] and [c,d] generate an affine map that maps
  // a -> c and b ->d and apply this mapping to n, return 
  // the resulting mapped value
  
  double2 m;
  
  double d1 = fabs(a0.x - a1.x)*fabs(a0.x - a1.x) + fabs(a0.y - a1.y)*fabs(a0.y - a1.y);
  double d2 = ((a0.x - a1.x)*(a0.x - a1.x) + (a0.y - a1.y)*(a0.y - a1.y))*
                (fabs(b0.x - b1.x)*fabs(b0.x - b1.x) + fabs(b0.y - b1.y)*fabs(b0.y - b1.y));
  
  m.x = a0.x - ((-(a1.x*((b0.x - b1.x)*(b0.x - n.x) + (b0.y - b1.y)*(b0.y - n.y))) + 
        a0.x*(b0.x*b0.x + b1.x*n.x - b0.x*(b1.x + n.x) + (b0.y - b1.y)*(b0.y - n.y)) + 
        (a0.y - a1.y)*(b0.y*b1.x - b0.x*b1.y - b0.y*n.x + b1.y*n.x + b0.x*n.y - b1.x*n.y))*d1)/d2;
  
  m.y = a0.y - ((a0.y*(b0.x*b0.x + b1.x*n.x - b0.x*(b1.x + n.x) + (b0.y - b1.y)*(b0.y - n.y)) - 
        a1.y*(b0.x*b0.x + b1.x*n.x - b0.x*(b1.x + n.x) + (b0.y - b1.y)*(b0.y - n.y)) + 
        (a0.x - a1.x)*(b0.x*b1.y - b1.y*n.x + b0.y*(-b1.x + n.x) - b0.x*n.y + b1.x*n.y))*d1)/d2;
  
  return m;
}

double2 orientation(double2 a0, double2 a1, double2 b0, double2 b1, double2 n)
{
  // given [a,b] and [c,d] generate an affine map that maps
  // a -> c and b ->d and apply this mapping to n, return 
  // the resulting mapped value
  
  double2 m;
  
  double d1 = sqrt((fabs(a0.x - a1.x)*fabs(a0.x - a1.x) + 
      fabs(a0.y - a1.y)*fabs(a0.y - a1.y))/(fabs(b0.x - b1.x)*fabs(b0.x - b1.x) 
      + fabs(b0.y - b1.y)*fabs(b0.y - b1.y)));
      
  double d2 = (a0.x - a1.x)*(a0.x - a1.x) + (a0.y - a1.y)*(a0.y - a1.y);
  
  m.x = ((((a0.x - a1.x)*(b0.x - b1.x) + (a0.y - a1.y)*(b0.y - b1.y))*n.x + 
      (-((a0.y - a1.y)*(b0.x - b1.x)) + (a0.x - a1.x)*(b0.y - b1.y))*n.y)*d1)/d2;
     
  m.y = ((((a0.y - a1.y)*(b0.x - b1.x) - (a0.x - a1.x)*(b0.y - b1.y))*n.x + 
      ((a0.x - a1.x)*(b0.x - b1.x) + (a0.y - a1.y)*(b0.y - b1.y))*n.y)*d1)/d2;
  
  return m;
}

double stickingProbability(double2 uv, double T)
{
  // return the probability of a particle sticking to a surface
  // given the velocity (x-> normal to surface, y -> tangential to 
  // surface) and temperature
  
  double normal = 1.0 - BETA_N*exp(-uv.x*uv.x/T);
  double tangential = 1.0 - BETA_T*(1.0 - exp(-uv.y*uv.y/T));
  
  return  normal*tangential;
}

