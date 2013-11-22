/////////////////////////////////////////
// rot90
/////////////////////////////////////////

void rot90(int* i, int* j, int n, int Ni, int Nj)
{
    //rotate a matrix index [i,j] by 90deg CCW n times
    // NOTE: N is equal to number of elements
    for (int a = 0; a < n; a++) {
        int temp = (*i);
        (*i) = Nj - (*j) - 1;
        (*j) = temp;
        temp = Ni;
        Ni = Nj;
        Nj = temp;
    }
}

/////////////////////////////////////////
// flip
/////////////////////////////////////////

void fliplr(int* i, int* j, int Ni, int Nj)
{
    //flip a matrix left to right
    // NOTE: N is equal to number of elements minus one
    (*i) = Ni - (*i);
}

void flipud(int* i, int* j, int Ni, int Nj)
{
    //flip a matrix left to right
    // NOTE: N is equal to number of elements minus one
    (*j) = Nj - (*j);
}

/////////////////////////////////////////
// getSwapIndex
/////////////////////////////////////////
    
void getSwapIndex(int this_face, int* iB, int* jB, int that_face, int NIB, int NJB, int v) 
{
    //get the corresponding index to i, j for ghost cell information swapping
    // achieves this through rotation of matrix A (this block) and then shifting of matrix A
    //    to align the ghost cells with the corresponding cells of matrix B (other block)
    // NOTE: "v" is used to distinguish between when we are looking for vertex indices or cell indices
    //         vertex -> v=1
    //         cell   -> v=0
  
  int adjust_flip, adjust_rot;
  if (v == 0) {
    adjust_flip = -1;
    adjust_rot = 0;
  } else {
    adjust_flip = 0;
    adjust_rot = 1;
  }
	
  if (this_face == GEAST) {
          if (that_face == GEAST) {
            fliplr(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*iB) += NIB - 2*GHOST;
          } else if (that_face == GWEST) {
            (*iB) -= NI - 2*GHOST;
          } else if (that_face == GNORTH) {
            rot90(iB,jB,1,NI+adjust_rot,NJ+adjust_rot);
            (*jB) += NJB - 2*GHOST;
          } else if (that_face == GSOUTH) {
            rot90(iB,jB,1,NIB+adjust_rot,NJB+adjust_rot);
            flipud(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*jB) -= NJ - 2*GHOST;
          }
        } else if (this_face == GWEST) {
          if (that_face == GWEST) {
            fliplr(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*iB) -= NI - 2*GHOST;
          } else if (that_face == GEAST) {
            (*iB) += NIB - 2*GHOST;
          } else if(that_face == GNORTH) {
            rot90(iB,jB,1,NI+adjust_rot,NJ+adjust_rot);
            fliplr(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*jB) += NJB - 2*GHOST;
          } else if (that_face == GSOUTH) {
            rot90(iB,jB,3,NI+adjust_rot,NJ+adjust_rot);
            (*jB) -= NJ - 2*GHOST;
          }
        } else if (this_face == GNORTH) {
          if (that_face == GNORTH) {
            flipud(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*jB) += NJB - 2*GHOST;
          } else if (that_face == GSOUTH) {
            (*jB) -= NJ - 2*GHOST;
          } else if (that_face == GEAST) {
            rot90(iB,jB,1,NI+adjust_rot,NJ+adjust_rot);
            (*iB) += NIB - 2*GHOST;
          } else if (that_face == GWEST) {
            rot90(iB,jB,1,NI+adjust_rot,NJ+adjust_rot);
            fliplr(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*iB) -= NI - 2*GHOST;
          } 
        } else if (this_face == GSOUTH) {
          if (that_face == GSOUTH) {
            flipud(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*jB) -= NJ - 2*GHOST;
          } else if (that_face == GNORTH) {
            (*jB) += NJB - 2*GHOST;
          } else if (that_face == GEAST) {
            rot90(iB,jB,3,NI+adjust_rot,NJ+adjust_rot);
            flipud(iB,jB,NI+adjust_flip,NJ+adjust_flip);
            (*iB) += NIB - 2*GHOST;
          } else if (that_face == GWEST) {
            rot90(iB,jB,1,NI+adjust_rot,NJ+adjust_rot);
            (*iB) -= NI - 2*GHOST;
          }
        }
  
  
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: xyExchange: block coords communication
////////////////////////////////////////////////////////////////////////////////
    
#define XYA(i,j) xyA[(i)*NY + (j)]
#define XYB(i,j) xyB[(i)*(NJB+1) + (j)]
    
__kernel void
xyExchange(__global double2* xyA,
	   int this_face,
	   __global double2* xyB,
	   int NIB, int NJB, int that_face,
     int ori)
{
  // update ghost cells
  
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  
  int A1i, A1j, A2i, A2j; // first and second coordinate indices
  int B1i, B1j, B2i, B2j;
  
  // get the top/bottom or left/right node indexes of this cells face
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST + 1;
      A1i = GHOST;
      A1j = NJ - GHOST;
      A2i = NI- GHOST;
      A2j = NJ - GHOST;
      break;
    case GEAST:
      gi += NI - GHOST + 1;
      gj += GHOST;
      A1i = NI - GHOST;
      A1j = GHOST;
      A2i = NI - GHOST;
      A2j = NJ - GHOST;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      A1i = GHOST;
      A1j = GHOST;
      A2i = NI - GHOST;
      A2j = GHOST;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      A1i = GHOST;
      A1j = GHOST;
      A2i = GHOST;
      A2j = NJ - GHOST;
      break;
  }
  
  // get the top/bottom or left/right node indexes of the other cells face
  switch (that_face) {
    case GNORTH:
      B1i = GHOST;
      B1j = NJB - GHOST;
      B2i = NIB- GHOST;
      B2j = NJB - GHOST;
      break;
    case GEAST:
      B1i = NIB - GHOST;
      B1j = GHOST;
      B2i = NIB - GHOST;
      B2j = NJB - GHOST;
      break;
    case GSOUTH:
      B1i = GHOST;
      B1j = GHOST;
      B2i = NIB - GHOST;
      B2j = GHOST;
      break;
    case GWEST:
      B1i = GHOST;
      B1j = GHOST;
      B2i = GHOST;
      B2j = NJB - GHOST;
      break;
  }
  
  double2 A1, A2, B1, B2; // first and second coordinates of the corner vertices of the given faces for block A and B
  
  A1 = XYA(A1i,A1j);
  A2 = XYA(A2i,A2j);
  
  B1 = XYB(B1i,B1j);
  B2 = XYB(B2i,B2j);
  
  int misalign = (!equal(A1.x,B1.x,1e-6)) || (!equal(A1.y,B1.y,1e-6)) || (!equal(A2.x,B2.x,1e-6)) || (!equal(A2.y,B2.y,1e-6));
  
  int iB, jB;
  
  for (int ii=0; ii<2; ++ii) {
    for (int jj=0; jj<2; ++jj) {
      iB = gi + ii;
      jB = gj + jj;
  
      // now align vertices on the edge
      if (misalign && ori) {
        // we are not aligned
        if ((this_face == GEAST) || (this_face == GWEST)) {
          flipud(&iB,&jB,NI,NJ);
        } else if ((this_face == GNORTH) || (this_face == GSOUTH)) {
          fliplr(&iB,&jB,NI,NJ);  // CHECK
        }
      }
      
      getSwapIndex(this_face, &iB, &jB, that_face, NIB, NJB, 1);
        
      XYA(gi+ii,gj+jj) = XYB(iB,jB);
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: xyExtrapolate: extrapolate face information
////////////////////////////////////////////////////////////////////////////////

__kernel void
xyExtrapolate(__global double2* xy, int this_face)
{
  // extrapolate information on face to give zero gradient
  // performed by extending last vector made by nodes of cell
  
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  
  // extrapolate index
  size_t ei, ej;
  
  double2 dxy1, dxy2;
  double m;
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      
      ei = gi;
      ej = NJ - GHOST - 1;
      
      m = (double)(gj - ej);	// multiplier, number of cells outside domain
      
      // spacing of previous (inside) cell
      
      dxy1 = m*(XY(ei,ej+1) - XY(ei,ej));
      
      dxy2 = m*(XY(ei+1,ej+1) - XY(ei+1,ej));
      
      XY(gi,gj) = XY(ei,ej) + dxy1;
      XY(gi+1,gj) = XY(ei+1,ej) + dxy2;
      
      XY(gi+1,gj+1) = XY(ei+1,ej+1) + dxy2;
      XY(gi,gj+1) = XY(ei,ej+1) + dxy1;
      
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      
      ei = NI - GHOST - 1;
      ej = gj;
      
      m = (double)(gi - ei);
      
      dxy1 = m*(XY(ei+1,ej) - XY(ei,ej));
      
      dxy2 = m*(XY(ei+1,ej+1) - XY(ei,ej+1));
      
      XY(gi,gj) = XY(ei,ej) + dxy1;
      XY(gi+1,gj) = XY(ei+1,ej) + dxy1;
      
      XY(gi+1,gj+1) = XY(ei+1,ej+1) + dxy2;
      XY(gi,gj+1) = XY(ei,ej+1) + dxy2;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      
      ei = gi;
      ej = GHOST;
      
      m = (double)(ej - gj);
      
      dxy1 = m*(XY(ei,ej) - XY(ei,ej+1));
      dxy2 = m*(XY(ei+1,ej) - XY(ei+1,ej+1));
      
      XY(gi,gj) = XY(ei,ej) + dxy1;
      XY(gi+1,gj) = XY(ei+1,ej) + dxy2;
      
      XY(gi+1,gj+1) = XY(ei+1,ej+1) + dxy2;
      XY(gi,gj+1) = XY(ei,ej+1) + dxy1;
      
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      
      ei = GHOST;
      ej = gj;
      
      m = (double)(ei - gi);
      
      dxy1 = m*(XY(ei,ej) - XY(ei+1,ej));
      dxy2 = m*(XY(ei,ej+1) - XY(ei+1,ej+1));
      
      XY(gi,gj) = XY(ei,ej) + dxy1;
      XY(gi+1,gj) = XY(ei+1,ej) + dxy1;
      
      XY(gi+1,gj+1) = XY(ei+1,ej+1) + dxy2;
      XY(gi,gj+1) = XY(ei,ej+1) + dxy2;
      
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeExchange: block communication
////////////////////////////////////////////////////////////////////////////////
    
#define fA(i,j,v) fA_[NV*NJ*(i) + NV*(j) + (v)] 
#define fB(i,j,v) fB_[NV*NJB*(i) + NV*(j) + (v)]
    
__kernel void
edgeExchange(__global double2* fA_,
             __global double2* xyA,
             int this_face,
             __global double2* fB_,
             __global double2* xyB,
             int NIB, int NJB, int that_face, 
             int flip, int ori)
{
  // update ghost cells
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  int A1i, A1j, A2i, A2j; // first and second coordinate indices
  int B1i, B1j, B2i, B2j;
  
   // get the top/bottom or left/right node indexes of this cells face
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      A1i = GHOST;
      A1j = NJ - GHOST;
      A2i = NI- GHOST;
      A2j = NJ - GHOST;
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      A1i = NI - GHOST;
      A1j = GHOST;
      A2i = NI - GHOST;
      A2j = NJ - GHOST;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      A1i = GHOST;
      A1j = GHOST;
      A2i = NI - GHOST;
      A2j = GHOST;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      A1i = GHOST;
      A1j = GHOST;
      A2i = GHOST;
      A2j = NJ - GHOST;
      break;
  }
  
  // get the top/bottom or left/right node indexes of the other cells face
  switch (that_face) {
    case GNORTH:
      B1i = GHOST;
      B1j = NJB - GHOST;
      B2i = NIB- GHOST;
      B2j = NJB - GHOST;
      break;
    case GEAST:
      B1i = NIB - GHOST;
      B1j = GHOST;
      B2i = NIB - GHOST;
      B2j = NJB - GHOST;
      break;
    case GSOUTH:
      B1i = GHOST;
      B1j = GHOST;
      B2i = NIB - GHOST;
      B2j = GHOST;
      break;
    case GWEST:
      B1i = GHOST;
      B1j = GHOST;
      B2i = GHOST;
      B2j = NJB - GHOST;
      break;
  }
  
  
  double2 A1, A2, B1, B2; // first and second coordinates of the corner vertices of the given faces for block A and B
  
  A1 = XYA(A1i,A1j);
  A2 = XYA(A2i,A2j);
  
  B1 = XYB(B1i,B1j);
  B2 = XYB(B2i,B2j);
  
  int misalign = (!equal(A1.x,B1.x,1e-6)) || (!equal(A1.y,B1.y,1e-6)) || (!equal(A2.x,B2.x,1e-6)) || (!equal(A2.y,B2.y,1e-6));
  
  int iB = gi;
  int jB = gj;
  
  // now align vertices on the edge
  if (misalign && ori) {
    // we are not aligned
    if ((this_face == GEAST) || (this_face == GWEST)) {
      flipud(&iB,&jB,NI-1,NJ-1);
    } else if ((this_face == GNORTH) || (this_face == GSOUTH)) {
      fliplr(&iB,&jB,NI-1,NJ-1);  // CHECK
    }
  }
  
  getSwapIndex(this_face, &iB, &jB, that_face, NIB, NJB, 0);
  
  
  size_t gv1, gv2;
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    gv1 = li*LOCAL_SIZE+ti;
    if (gv1 < NV) {
      
      // THIS NEEDS VERIFICATION FOR ALL CASES
      if (flip == NO_FLIP) {
        gv2 = gv1;
      } else if (flip == FLIP_NS) {
        gv2 = mirror_NS[gv1];
      } else if (flip == FLIP_D) {
        gv2 = mirror_D[gv1];
      } else if (flip == FLIP_EW) {
        gv2 = mirror_EW[gv1];
      }
      
      fA(gi,gj,gv1) = fB(iB,jB,gv2);
    }
  }
      
  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeExtrapolate: extrapolate face information
////////////////////////////////////////////////////////////////////////////////

__kernel void
edgeExtrapolate(__global double2* Fin,
	   int this_face)
{
  // extrapolate information on face to give zero gradient
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  // extrapolate index
  size_t ei, ej;
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      
      ei = gi;
      ej = NJ - GHOST - 1;
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      
      ei = NI - GHOST - 1;
      ej = gj;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      
      ei = gi;
      ej = GHOST;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      
      ei = GHOST;
      ej = gj;
      break;
  }
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      F(gi,gj,gv) = F(ei,ej,gv);
    }
  }
  
  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeConstant: set constant edge data
////////////////////////////////////////////////////////////////////////////////

__kernel void
edgeConstant(__global double2* Fin,
	   int this_face,
	   __global double4* wall_prop)
{
  // set distribution functions to the equilibrium value defined by the input data
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  size_t ci;
  
  switch (this_face) {
    case GNORTH:
      ci = gi;
      gi += GHOST;
      gj += NJ - GHOST;
      break;
    case GEAST:
      ci = gj;
      gi += NI - GHOST;
      gj += GHOST;
      break;
    case GSOUTH:
      ci = gi;
      gi += GHOST;
      gj += 0;
      break;
    case GWEST:
      ci = gj;
      gi += 0;
      gj += GHOST;
      break;
  }

  double2 uv;
  
  double4 prim;
  double2 Q;
  
  prim = WALL_PROP(this_face, ci);
  
  Q = 0.0;
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      uv = QUAD[gv];
      F(gi,gj,gv) = fEQ(prim, Q, uv, gv);
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeInflow: define the flow into the domain
////////////////////////////////////////////////////////////////////////////////

__kernel void
edgeInflow(__global double2* Fin,
	   int this_face,
     __global double2* normal,
	   __global double4* macro,
     __global double2* gQ,
     __global double4* wall_prop)
{
  // set distribution functions to the equilibrium value defined by the input data
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  // extrapolate index
  size_t ei, ej, ci, noi, noj, face;
  
  switch (this_face) {
    case GNORTH:
      ci = gi;
      gi += GHOST;
      gj += NJ - GHOST;
      ei = gi;
      ej = NJ - GHOST - 1;
      noi = gi;
      noj = gj;
      face = SOUTH;
      break;
    case GEAST:
      ci = gj;
      gi += NI - GHOST;
      gj += GHOST;
      ei = NI - GHOST - 1;
      ej = gj;
      noi = gi;
      noj = gj;
      face = WEST;
      break;
    case GSOUTH:
      ci = gi;
      gi += GHOST;
      gj += 0;
      ei = gi;
      ej = GHOST;
      noi = ei;
      noj = ej;
      face = SOUTH;
      break;
    case GWEST:
      ci = gj;
      gi += 0;
      gj += GHOST;
      ei = GHOST;
      ej = gj;
      noi = ei;
      noj = ej;
      face = WEST;
      break;
  }

  double4 prim = WALL_PROP(this_face, ci);
  double2 Q = GQ(ei-GHOST,ej-GHOST);
  double4 ghost_macro = MACRO(ei-GHOST,ej-GHOST);
  
  // the wall normal
  double2 wall_normal = NORMAL(noi,noj,face);
  
  // calculate the velocity normal to the interface
  ghost_macro.s12 = toLocal(ghost_macro.s12, wall_normal);
  
  // set the density and temperature
  ghost_macro.s0 = prim.s0;
  ghost_macro.s3 = prim.s3; // note that this temperature is actually 1/T
  
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      double2 uv = QUAD[gv];
      F(gi,gj,gv) = fEQ(ghost_macro, Q, uv, gv);
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeOutflow: define the flow out of the domain
////////////////////////////////////////////////////////////////////////////////

__kernel void
edgeOutflow(__global double2* Fin,
	   int this_face,
     __global double2* normal,
	   __global double4* macro,
     __global double2* gQ,
     __global double4* wall_prop)
{
  // set distribution functions to the equilibrium value defined by the input data
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  // extrapolate index
  size_t ei, ej, ci, noi, noj, face;
  
  switch (this_face) {
    case GNORTH:
      ci = gi;
      gi += GHOST;
      gj += NJ - GHOST;
      ei = gi;
      ej = NJ - GHOST - 1;
      noi = gi;
      noj = gj;
      face = SOUTH;
      break;
    case GEAST:
      ci = gj;
      gi += NI - GHOST;
      gj += GHOST;
      ei = NI - GHOST - 1;
      ej = gj;
      noi = gi;
      noj = gj;
      face = WEST;
      break;
    case GSOUTH:
      ci = gi;
      gi += GHOST;
      gj += 0;
      ei = gi;
      ej = GHOST;
      noi = ei;
      noj = ej;
      face = SOUTH;
      break;
    case GWEST:
      ci = gj;
      gi += 0;
      gj += GHOST;
      ei = GHOST;
      ej = gj;
      noi = ei;
      noj = ej;
      face = WEST;
      break;
  }

  double4 prim = WALL_PROP(this_face, ci);
  double2 Q = GQ(ei-GHOST,ej-GHOST);
  double4 ghost_macro = MACRO(ei-GHOST,ej-GHOST);
  
  // the wall normal
  double2 wall_normal = NORMAL(noi, noj, face);
  
  // calculate the velocity normal to the interface
  ghost_macro.s12 = toLocal(ghost_macro.s12, wall_normal);
  
  // set the temperature so that we have the right pressure
  ghost_macro.s3 = ghost_macro.s0/(prim.s0/prim.s3);
  
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      double2 uv = QUAD[gv];
      F(gi,gj,gv) = fEQ(ghost_macro, Q, uv, gv);
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeMirror: mirror face information
////////////////////////////////////////////////////////////////////////////////

__kernel void
edgeMirror(__global double2* Fin,
	   int this_face)
{
  // mirror edge
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  // mirrored index
  size_t mi, mj, yes;
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      mi = gi;
      mj = 2*(NJ - GHOST) - gj - 1;
      yes = 1;
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      mi = 2*(NI - GHOST) - gi - 1;
      mj = gj;
      yes = 2;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      mi = gi;
      mj = 2*GHOST - gj - 1;
      yes = 1;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      mi = 2*GHOST - gi - 1;
      mj = gj;
      yes = 2;
      break;
  }
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      if (yes == 1) {
        int mv = mirror_NS[gv];
        F(gi,gj,gv) = F(mi,mj,mv);
      }
      else if (yes == 2) {
        int mv = mirror_EW[gv];
        F(gi,gj,gv) = F(mi,mj,mv);
      }
    }
  }
  
  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeBounceBack: double mirror face information
////////////////////////////////////////////////////////////////////////////////

__kernel void
edgeBounceBack(__global double2* Fin,
	   int this_face)
{
  // mirror edge
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  // mirrored index
  size_t mi, mj;
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      mi = gi;
      mj = 2*(NJ - GHOST) - gj - 1;
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      mi = 2*(NI - GHOST) - gi - 1;
      mj = gj;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      mi = gi;
      mj = 2*GHOST - gj - 1;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      mi = 2*GHOST - gi - 1;
      mj = gj;
      break;
  }
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      int mv = mirror_D[gv];
      F(gi,gj,gv) = F(mi,mj,mv);
    }
  }
  
  return;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: edgeConstGrad: extrapolate face information
////////////////////////////////////////////////////////////////////////////////

__kernel void
edgeConstGrad(__global double2* Fin, int this_face)
{
  // extrapolate information on face to give constant gradient
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  // extrapolate index
  int ei0, ej0, ei1, ej1;
  
  double m;
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      
      ei1 = gi;
      ej1 = NJ - GHOST - 1;
      
      ei0 = ei1;
      ej0 = ej1 - 1;
      
      m = (double)(gj - ej1);
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      
      ei1 = NI - GHOST - 1;
      ej1 = gj;
      
      ei0 = ei1 - 1;
      ej0 = ej1;
      
      m = (double)(gi - ei1);
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      
      ei1 = gi;
      ej1 = GHOST;
      
      ei0 = ei1;
      ej0 = ej1 + 1;
      
      m = (double)(ej1 - gj);
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      
      ei1 = GHOST;
      ej1 = gj;
      
      ei0 = ei1 + 1;
      ej0 = ej1;
      
      m = (double)(ei1 - gi);
      break;
  }
  
  double2 f0, f1, df;
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      
      f0 = F(ei0,ej0,gv);
      f1 = F(ei1,ej1,gv);
      df = f1 - f0;
      
      F(gi,gj,gv) = f1 + m*df;
    }
  }

  return;
}
