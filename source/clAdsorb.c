
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

double vartheta_langmuir(double D, double T, size_t face)
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
    
    double vartheta = interpolate(pt, ISO[tri.x], ISO[tri.y], ISO[tri.z]);
    
    //printf("vartheta = %g\n",vartheta);
    
    return vartheta;
}
