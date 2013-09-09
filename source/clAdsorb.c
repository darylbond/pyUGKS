
// All the functions required for implementation of the adsorbing boundary condition
#define MAXITS 100
#define SGN(x) (((x) > 0.0) - ((x) < 0.0))

double stickingProbability(double2 uv, double T, size_t face)
{
  // return the probability of a particle sticking to a surface
  // given the velocity (x-> normal to surface, y -> tangential to 
  // surface) and temperature
  
  double normal = 1.0 - BETA_N[face]*exp(-uv.x*uv.x/T);
  double tangential = 1.0 - BETA_T[face]*(1.0 - exp(-uv.y*uv.y/T));
  
  return  normal*tangential;
}

double distance_from_segment(double2 a, double2 b, double2 c)
{
	double r_numerator = (c.x-a.x)*(b.x-a.x) + (c.y-a.y)*(b.y-a.y);
	double r_denomenator = (b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y);
	double r = r_numerator / r_denomenator;
    //
    double px = a.x + r*(b.x-a.x);
    double py = a.y + r*(b.y-a.y);
    //     
    double s =  ((a.y-c.y)*(b.x-a.x)-(a.x-c.x)*(b.y-a.y) ) / r_denomenator;

	double distanceLine = fabs(s)*sqrt(r_denomenator);
    double distanceSegment;

	if ( (r >= 0) && (r <= 1) )
	{
		distanceSegment = distanceLine;
	}
	else
	{
		double dist1 = (c.x-a.x)*(c.x-a.x) + (c.y-a.y)*(c.y-a.y);
		double dist2 = (c.x-b.x)*(c.x-b.x) + (c.y-b.y)*(c.y-b.y);
		if (dist1 < dist2)
		{
			distanceSegment = sqrt(dist1);
		}
		else
		{
			distanceSegment = sqrt(dist2);
		}
	}
	return distanceSegment;
}

int intersect(double2 a, double2 b, double2 c, double2 d)
{
    // line segment intersection
    float den = ((d.y-c.y)*(b.x-a.x)-(d.x-c.x)*(b.y-a.y));
    float num1 = ((d.x - c.x)*(a.y-c.y) - (d.y- c.y)*(a.x-c.x));
    float num2 = ((b.x-a.x)*(a.y-c.y)-(b.y-a.y)*(a.x-c.x));
    float u1 = num1/den;
    float u2 = num2/den;
    if (den == 0 && num1  == 0 && num2 == 0)
        /* The two lines are coincidents */
        return 0;
    if (den == 0)
        /* The two lines are parallel */
        return 0;
    if (u1 <0 || u1 > 1 || u2 < 0 || u2 > 1)
        /* Lines do not collide */
        return 0;
    /* Lines DO collide */
    return 1;
}

int2 get_tri(double2 xy,
            __constant double4* pts,
            __constant int4* tris,
            int size_tri,
            __constant int4* nbrs,
            int this_tri_id)
{
    // return the tri index that contains the input point, and whether 
    // it IS the tri, or just the last one before we went outside the domain
    
    double2 a, b, c;
    int3 hit;
    int outside = 0;
    double tol = 1e-12;
    for(size_t counter = 0; counter < size_tri; ++counter) { //N_TRI
        // we are at the point start and want to get to xy
        // draw a line from start to xy and see which face of this_tri
        // the line intersects
        
        int3 this_tri = tris[this_tri_id].s012;
        
        double2 start = (1.0/3.0)*(pts[this_tri.x].s01 + pts[this_tri.y].s01 + pts[this_tri.z].s01);

        a = pts[this_tri.x].s01;
        b = pts[this_tri.y].s01;
        c = pts[this_tri.z].s01;
        
        // face 1
        hit.z = intersect(a, b, start, xy);
        
        // face 2
        hit.x = intersect(b, c, start, xy);
        
        // face 3
        hit.y = intersect(c, a, start, xy);
        
        if ((hit.x + hit.y + hit.z) == 0.0) {
            // the point xy is inside this triangle
            break;
        }
        
        // have hit a face of the triangle and now we want to know 
        // the triangle that this face leads to
        
        int3 nbr = nbrs[this_tri_id].s012;
    
        if (hit.x) {
            if (nbr.x == -1) {
                // the point is outside the triangulation
                // we have gotten as close as we can
                outside = 1;
                break;
            } else if (distance_from_segment(b, c, xy) < tol) {
                // the point is on the edge
                break;
            } else {
                this_tri_id = nbr.x;
            }
        } else if (hit.y) {
            if (nbr.y == -1) {
                outside = 1;
                break;
            } else if (distance_from_segment(c, a, xy) < tol) {
                break;
            } else {
                this_tri_id = nbr.y;
            }
        } else if (hit.z) {
            if (nbr.z == -1) {
                outside = 1;
                break;
            }else if (distance_from_segment(a, b, xy) < tol) {
                break;
            } else {
                this_tri_id = nbr.z;
            }
        }
    }
    
    int2 out;
    out.x = this_tri_id;
    out.y = outside;
    
    return out;
}

double interpolate(double2 pt, double4 a, double4 b, double4 c)
{
    // interpolate the value at a point in the provided triangle
  
    // barycentric interpolation
    double L1 = ((b.y-c.y)*(pt.x-c.x)+(c.x-b.x)*(pt.y-c.y))/((b.y-c.y)*(a.x-c.x)+(c.x-b.x)*(a.y-c.y));
    double L2 = ((c.y-a.y)*(pt.x-c.x)+(a.x-c.x)*(pt.y-c.y))/((b.y-c.y)*(a.x-c.x)+(c.x-b.x)*(a.y-c.y));
    double L3 = 1 - L1 - L2;
    
    return L1*a.z + L2*b.z + L3*c.z;
}

double2 vartheta_langmuir(double D, double T, size_t face)
{
    // given the density and temperature find the equilibrium isotherm 
    // for the given face
  
    double2 pt = (double2)(0.5*D*T, T); // (pressure, temperature)
  
    // do some interpolation to get a vartheta value

    __constant double4* ISO;
    __constant int4* TRI;
    __constant int4* NBR;


    switch (face) {
        case GNORTH:
            ISO = ISO_north;
            TRI = TRI_north;
            NBR = NBR_north;
            break;
        case GEAST:
            ISO = ISO_east;
            TRI = TRI_east;
            NBR = NBR_east;
            break;
        case GSOUTH:
            ISO = ISO_south;
            TRI = TRI_south;
            NBR = NBR_south;
            break;
        case GWEST:
            ISO = ISO_west;
            TRI = TRI_west;
            NBR = NBR_west;
            break;
    }

    int2 loc = get_tri(pt, ISO, TRI, N_TRI[face], NBR, 0);
    
    //if (loc.y) {
        //// we are outside the interpolation range, flag an error!!
        //printf("OUTSIDE RANGE, P = %g, T = %g\n",pt.x, pt.y);
    //}

    int3 tri = TRI[loc.x].s012;
    
    double2 vartheta;
    vartheta.x = interpolate(pt, ISO[tri.x], ISO[tri.y], ISO[tri.z]);
    vartheta.y = (double) loc.y;
    
    return vartheta;
}

#if (HAS_CL_WALL || HAS_SPECULAR_WALL)  // adsorb_CL & adsorb_specular-diffuse
#define COVER(w, i) wall_cover[(i)*4 + (w)]
#define WALL_DIST(i, v) wall_dist[(i)*NV + (v)]

#define GOLDEN_RATIO 0.618

__kernel void
adsorbingWall_P1(__global double2* normal,
                    int face,
                    __global double4* wall_prop,
                    __global double4* wall_cover,
                    __global double2* wall_dist,
                    __global double2* flux_f,
                    __global double4* macro,
                    double dt,
                    __global double* flag)
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
    double T, D;

    switch (face) {
        case GNORTH:
            ci = gi;
            gi += GHOST;
            gj += NJ - GHOST;
            rot = -1;
            face_id = SOUTH;
            T = 1.0/MACRO(ci, gj-GHOST-1).s3;
            D = MACRO(ci, gj-GHOST-1).s0;
            break;
        case GEAST:
            ci = gj;
            gi += NI - GHOST;
            gj += GHOST;
            rot = -1;
            face_id = WEST;
            T = 1.0/MACRO(gi-GHOST-1, ci).s3;
            D = MACRO(gi-GHOST-1, ci).s0;
            break;
        case GSOUTH:
            ci = gi;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = SOUTH;
            T = 1.0/MACRO(ci, gj-GHOST).s3;
            D = MACRO(ci, gj-GHOST).s0;
            break;
        case GWEST:
            ci = gj;
            gi += GHOST;
            gj += GHOST;
            rot = 1;
            face_id = WEST;
            T = 1.0/MACRO(gi-GHOST, ci).s3;
            D = MACRO(gi-GHOST, ci).s0;
            break;
    }
    
    __local double2 data[LOCAL_SIZE];
    __local double adjusted_flux[LOCAL_SIZE];
    __local double adsorb_ratio[1];
    
    adsorb_ratio[0] = 0.0;

    if (((face_id == SOUTH) && ((gi - GHOST) < ni)) 
    || ((face_id == WEST) && ((gj - GHOST) < nj))) {

        // calculate the updated wall coverage

        double2 face_normal = NORMAL(gi,gj,face_id);
        
        double beta, dvtheta;
        double2 total_flux, reflected_flux, adsorbed_flux, desorbed_flux;
        double2 uv, face_dist;
        int delta, zero_dvtheta;
        
        //data = [reflected flux, adsorbed flux]
        
        data[thread_id] = 0.0;
        adjusted_flux[thread_id] = 0.0;
        
        double vartheta = COVER(face, ci).s0; // the fraction of cover that this wall section has
        
        for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
            size_t gv = li*LOCAL_SIZE+thread_id;
            if (gv < NV) {
                uv = interfaceVelocity(gv, face_normal);

                face_dist = FLUXF(gi,gj,gv); // dist. on the interface

                delta = (sign(uv.x)*rot + 1)/2; // flag to indicate if this velocity is going out of the wall (delta = 1 -> out of wall)
                
                total_flux = -rot*uv.x*(1-delta)*face_dist; // total flux that is impinging on wall
                
                beta = stickingProbability(uv, T, face); // probability giving the amount of this discrete velocity that has a chance of sticking 
                
                adsorbed_flux = beta*total_flux*GAMMA_F[face]*(1.0 - vartheta); // is adsorbed
                
                if (adsorbed_flux.x > total_flux.x) {
                    adsorbed_flux = total_flux;
                }
                
                reflected_flux = total_flux - adsorbed_flux; // is reflected
                
                
                // sum of the reflected and adsorbed distributions
                data[thread_id].x += reflected_flux.x;
                data[thread_id].y += adsorbed_flux.x;
                
                // the flux that is seen by the Langmuir reaction
                adjusted_flux[thread_id] += beta*total_flux.x;
                
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
                adjusted_flux[thread_id] += adjusted_flux[thread_id+step];
            }
            step *= 2;
            grab_id *= 2;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (thread_id == 0) {
        
            // we now know how much needs to be reflected and adsorbed
            
            reflected_flux.x = data[0].x;
            adsorbed_flux.x  = data[0].y;
            
            zero_dvtheta = 0;
            
            if (GAMMA_F[face] > 0.0) {
            
                dvtheta = dt*ALPHA_P[face]*adsorbed_flux.x;
                
                if (dvtheta > (1 - vartheta)) {
                    // we are adsorbing too much!
                    flag[0] = ERR_ADSORB_TOO_MUCH;
                    // calculate the time step required to adsorb only half as much as is 
                    // physically possible
                    // we now use this dt so that we can check the desorbing process as well
                    dt = GOLDEN_RATIO*(1 - vartheta)/(ALPHA_P[face]*adsorbed_flux.x);
                    flag[1] = dt;
                    
                    dvtheta = dt*ALPHA_P[face]*adsorbed_flux.x;
                    zero_dvtheta = 1;
                }
                
                // calculate the desorption rate based on the Langmuir isotherm
                double2 veq = vartheta_langmuir(D, T, face);
                
                if (veq.y == 1) {
                    // we are outside the bounds of the defined adsorption data
                    flag[0] = ERR_CLAMPING_ADSORB_ISOTHERM;
                    return;
                }
                
                double gamma_b = GAMMA_F[face]*(1.0/veq.x - 1.0)*adjusted_flux[0];
                
                desorbed_flux.x  = gamma_b*vartheta;
                
                dvtheta -= dt*ALPHA_P[face]*desorbed_flux.x;
                
                // make sure we don't get negative ratio of coverage
                // also ensure that we only desorb what is available, not 
                //  including what we have just adsorbed in this time step
                if ((vartheta + dvtheta) < 0.0) {
                    desorbed_flux.x = vartheta/(dt*ALPHA_P[face]);
                    dt = GOLDEN_RATIO*vartheta/(ALPHA_P[face]*desorbed_flux.x);
                    flag[0] = ERR_DESORB_TOO_MUCH;
                    flag[1] = dt;
                    zero_dvtheta = 1;
                }
            } else {
                dvtheta = 0.0;
            }
            
            if (zero_dvtheta == 0) {
                COVER(face, ci).s0 += dvtheta;
                COVER(face, ci).s1 = reflected_flux.x;
                COVER(face, ci).s2 = adsorbed_flux.x;
                COVER(face, ci).s3 = desorbed_flux.x;
            }
        }
    }
    
    return;
}

#endif

#if HAS_CL_WALL // adsorb_CL

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
                                CL = CercignaniLampis(uv_in, uv_out, WALL_DIST(ci, va), ALPHA_N[face], ALPHA_T[face], 1.0/wall.s3);
                                CL.y += 0.5*ALPHA_T[face]*(2-ALPHA_T[face])/wall.s3*CL.x;
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

#if HAS_SPECULAR_WALL // adsorb_specular-diffuse
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
