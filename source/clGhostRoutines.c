/////////////////////////////////////////
// rot90
/////////////////////////////////////////

void rot90(int* i, int* j, int n)
{
    //rotate a matrix index [i,j] by 90deg CCW n times
    int Ni = NI;
    int Nj = NJ;
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
// getSwapIndex
/////////////////////////////////////////
    
void getSwapIndex(int faceA, int* i, int* j, int faceB, int NIB, int NJB) 
{
    //get the corresponding index to i, j for ghost cell information swapping
    // achieves this through rotation of matrix A (this block) and then shifting of matrix A
    //    to align the ghost cells with the corresponding cells of matrix B (other block)
    
  if (faceA == GNORTH) {
        if (faceB == GNORTH) {
            rot90(i,j,2);
            (*j) = (*j) + NJB - 2*GHOST;
	}
	else if (faceB == GEAST) {
	  rot90(i,j,1);
	  (*i) = (*i) + NIB - 2*GHOST;
	}
        else if (faceB == GSOUTH) {
            rot90(i,j,0);
            (*j) = (*j) - NJ + 2*GHOST;
	}
        else if (faceB == GWEST) {
            rot90(i,j,3);
            (*i) = (*i) - NI + 2*GHOST;
	}
    }
    
    else if (faceA == GEAST) {
        if (faceB == GNORTH) {
            rot90(i,j,3);
            (*j) = (*j) + NJB - 2*GHOST;
	}
        else if (faceB == GEAST) {
            rot90(i,j,2);
            (*i) = (*i) + NIB - 2*GHOST;
	}
        else if (faceB == GSOUTH) {
            rot90(i,j,1);
            (*j) = (*j) - NJ + 2*GHOST;
	}
        else if (faceB == GWEST) {
            rot90(i,j,0);
            (*i) = (*i) - NI + 2*GHOST;
	}
    }
            
    else if (faceA == GSOUTH) {
        if (faceB == GNORTH) {
            rot90(i,j,0);
            (*j) = (*j) + NJB - 2*GHOST;
	}
        else if (faceB == GEAST) {
            rot90(i,j,3);
            (*i) = (*i) + NIB - 2*GHOST;
	}
        else if (faceB == GSOUTH) {
            rot90(i,j,2);
            (*j) = (*j) - NJ + 2*GHOST;
	}
        else if (faceB == GWEST) {
            rot90(i,j,1);
            (*i) = (*i) - NI + 2*GHOST;
	}
    }
        
    else if (faceA == GWEST) {
        if (faceB == GNORTH) {
            rot90(i,j,1);
            (*j) = (*j) + NJB - 2*GHOST;
	}
        else if (faceB == GEAST) {
            rot90(i,j,0);
            (*i) = (*i) + NIB - 2*GHOST;
	}
        else if (faceB == GSOUTH) {
            rot90(i,j,3);
            (*j) = (*j) - NJ + 2*GHOST;
	}
        else if (faceB == GWEST) {
            rot90(i,j,2);
            (*i) = (*i) - NI + 2*GHOST;
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
	   int NIB, int NJB, int that_face)
{
  // update ghost cells
  
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      break;
  }
  
  int iB = gi;
  int jB = gj;
  
  getSwapIndex(this_face, &iB, &jB, that_face, NIB, NJB);
  
  XYA(gi,gj) = XYB(iB,jB);
  XYA(gi+1,gj) = XYB(iB+1,jB);
  XYA(gi+1,gj+1) = XYB(iB+1,jB+1);
  XYA(gi,gj+1) = XYB(iB,jB+1);
      
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
	   int this_face,
	   __global double2* fB_,
	   int NIB, int NJB,int that_face)
{
  // update ghost cells
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      break;
  }
  
  int iB = gi;
  int jB = gj;
  
  getSwapIndex(this_face, &iB, &jB, that_face, NIB, NJB);
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      fA(gi,gj,gv) = fB(iB,jB,gv);
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
	   double D, double U, double V, double T)
{
  // set distribution functions to the equilibrium value defined by the input data
  
  // distribution functions global index
  size_t gi = get_global_id(0);
  size_t gj = get_global_id(1);
  size_t ti = get_local_id(2);
  
  switch (this_face) {
    case GNORTH:
      gi += GHOST;
      gj += NJ - GHOST;
      break;
    case GEAST:
      gi += NI - GHOST;
      gj += GHOST;
      break;
    case GSOUTH:
      gi += GHOST;
      gj += 0;
      break;
    case GWEST:
      gi += 0;
      gj += GHOST;
      break;
  }

  double2 uv;
  
  double4 prim;
  double2 Q;
  
  prim.s0 = D;
  prim.s1 = U;
  prim.s2 = V;
  prim.s3 = 1.0/T;
  Q = 0.0;
  
  for (size_t li = 0; li < LOCAL_LOOP_LENGTH; ++li) {
    size_t gv = li*LOCAL_SIZE+ti;
    if (gv < NV) {
      uv = QUAD[gv];
      F(gi,gj,gv) = fEQ(prim, Q, uv);
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
