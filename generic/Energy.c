#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Energy.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

long poten_(Energy_c_fact)(lua_State *L, long f){
  if ( f<2 ) {
    return 1;
  } else {
    return f * poten_(Energy_c_fact)(L, f-1);
  }
}

static int poten_(Energy_neighborPairUpdateAll)(lua_State *L)
{
  THTensor *coords_ptr = luaT_getfieldcheckudata(L, 1, "atomPositions", torch_Tensor);
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_syms = luaT_getfieldchecknumber(L, 1, "numSyms");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  real cutoff = luaT_getfieldchecknumber(L, 1, "potCutoff");
  real neighbin = luaT_getfieldchecknumber(L, 1, "neighborBin");
  THTensor * pairCutoffs_ptr = luaT_getfieldcheckudata(L, 1, "potPairCutoffs", torch_Tensor);
  THTensor * currentPartList_ptr = luaT_getfieldcheckudata(L, 1, "partList", torch_Tensor);
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * puv = luaT_getfieldcheckudata(L, 1, "unitCell", torch_Tensor);
  THTensor * numOfPartNeighList_ptr = luaT_getfieldcheckudata(L, 1, "numNeighList", torch_Tensor);
  THTensor * numOfPartNeighListPair_ptr = luaT_getfieldcheckudata(L, 1, "numNeighListPair", torch_Tensor);
  THTensor * neighListOfParts_ptr = luaT_getfieldcheckudata(L, 1, "partNeighList", torch_Tensor);
  THTensor * neighListOfPartsPair_ptr = luaT_getfieldcheckudata(L, 1, "partNeighListPair", torch_Tensor);
  THTensor * Rij_list_ptr = luaT_getfieldcheckudata(L, 1, "RijList", torch_Tensor);
  THTensor * Rij_list_Pair_ptr = luaT_getfieldcheckudata(L, 1, "RijListPair", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);

//  long pairs = num_typ + (long)(num_typ * (num_typ-1))/2;

//  real* input_data  = THTensor_(data)(input);
  real * currentPartList = THTensor_(data)(currentPartList_ptr);
  long   currentPartList_s = THTensor_(stride)(currentPartList_ptr, 0);
  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * coords = THTensor_(data)(coords_ptr);
  long   coords_s0 = THTensor_(stride)(coords_ptr, 0);
  long   coords_s1 = THTensor_(stride)(coords_ptr, 1);
  real * uv = THTensor_(data)(puv);
  long   uv_s0 = THTensor_(stride)(puv, 0);
  long   uv_s1 = THTensor_(stride)(puv, 1);
  real * uvp;
  
  real * pairCutoffs = THTensor_(data)(pairCutoffs_ptr);
  long   pairCutoffs_s = THTensor_(stride)(pairCutoffs_ptr, 0);

  THTensor_(resize1d)(numOfPartNeighList_ptr, partCounted);
  real * numOfPartNeighList  = THTensor_(data)(numOfPartNeighList_ptr);
  long   numOfPartNeighList_s = THTensor_(stride)(numOfPartNeighList_ptr, 0);
  THTensor_(resize2d)(numOfPartNeighListPair_ptr, partCounted, num_typ);
  THTensor_(zero)(numOfPartNeighListPair_ptr);
  real * numOfPartNeighListPair  = THTensor_(data)(numOfPartNeighListPair_ptr);
  long   numOfPartNeighListPair_s0 = THTensor_(stride)(numOfPartNeighListPair_ptr, 0);
  long   numOfPartNeighListPair_s1 = THTensor_(stride)(numOfPartNeighListPair_ptr, 1);

  THTensor_(resize2d)(neighListOfParts_ptr, partCounted, 2*partCounted);
  real * neighListOfParts  = THTensor_(data)(neighListOfParts_ptr);
  long   neighListOfParts_s0 = THTensor_(stride)(neighListOfParts_ptr, 0);
  long   neighListOfParts_s1 = THTensor_(stride)(neighListOfParts_ptr, 1);
  THTensor_(resize3d)(neighListOfPartsPair_ptr, partCounted, num_typ, 2*partCounted);
  real * neighListOfPartsPair  = THTensor_(data)(neighListOfPartsPair_ptr);
  long   neighListOfPartsPair_s0 = THTensor_(stride)(neighListOfPartsPair_ptr, 0);
  long   neighListOfPartsPair_s1 = THTensor_(stride)(neighListOfPartsPair_ptr, 1);
  long   neighListOfPartsPair_s2 = THTensor_(stride)(neighListOfPartsPair_ptr, 2);
  
  THTensor_(resize3d)(Rij_list_ptr, partCounted, 2*partCounted, 3);
  real * Rij_list  = THTensor_(data)(Rij_list_ptr);
  long   Rij_list_s0 = THTensor_(stride)(Rij_list_ptr, 0);
  long   Rij_list_s1 = THTensor_(stride)(Rij_list_ptr, 1);
  long   Rij_list_s2 = THTensor_(stride)(Rij_list_ptr, 2);
  THTensor_(resize4d)(Rij_list_Pair_ptr, partCounted, num_typ, 2*partCounted, 3);
  real * Rij_list_Pair  = THTensor_(data)(Rij_list_Pair_ptr);
  long   Rij_list_Pair_s0 = THTensor_(stride)(Rij_list_Pair_ptr, 0);
  long   Rij_list_Pair_s1 = THTensor_(stride)(Rij_list_Pair_ptr, 1);
  long   Rij_list_Pair_s2 = THTensor_(stride)(Rij_list_Pair_ptr, 2);
  long   Rij_list_Pair_s3 = THTensor_(stride)(Rij_list_Pair_ptr, 3);

  long i,j,k,d;
  long l,m,n;
//  long numOfPartNeigh;
  real Rsqij, Rsqik, Rsqjk;
  real cutsq = cutoff * cutoff; 
  if(neighbin > 0.0) cutsq = cutsq + (neighbin * neighbin);
  long offset;

  long* num_atom_typs = (long*) malloc(num_typ*sizeof(long));
  
  for(d = 0; d < num_typ; d++) {
      num_atom_typs[d]=0;
  }
  for (i = 0; i < partCounted; ++i) {
    for(d = 0; d < num_typ; d++) {
       if((long)particleSpecies[i * particleSpecies_s]-1 == d){
          num_atom_typs[d]++;
       }
    }
  }
  num_syms = num_typ * (num_rad + num_ang) + 
             ( poten_(Energy_c_fact)(L, num_typ) / ( 2 * poten_(Energy_c_fact)(L, num_typ-2))) * num_ang;
  
  THTensor_(resize1d)(PartListOffset_ptr, partCounted);
  real* PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long  PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);
    
  long*  specieCount = (long *) malloc(num_typ*sizeof(long));
  long* specieOffset = (long *) malloc((num_typ+1)*sizeof(long));
  
  specieOffset[0]=0;
  for (d=0;d<num_typ;d++){
        specieCount[d]=0;
        if (d<1) { // Second specie's offset
          specieOffset[d+1] = num_atom_typs[d];
        } else { // Shift others accordingly
          specieOffset[d+1] = specieOffset[d] + num_atom_typs[d];
        }
  }

  // Calculate neighbor lists over all atoms
 
 //real scaledVec[27][3];
 real scaledVec[125][3];
 i=0;
 for(l=-2;l<=2;l++){
   for(m=-2;m<=2;m++){
     for(n=-2;n<=2;n++){
         for(d=0;d<3;d++) { 
            uvp = uv + d * uv_s1;
            scaledVec[i][d] = l * uvp[0] + m * uvp[uv_s0] + n * uvp[2 * uv_s0];
         }
         i++;
     }
   }
 }

//    real * cip;
//    real * cjp;
    long  ityp;
//    long dd,ll;
//    real o[3],t[3];
//    real dij;
//    long numOfPartNeigh=0;
//    real *partNLp;
//    real *Rijpi;
//    real *Rijpj;

 for(i=0;i<partCounted;i++){
   ityp =  (long)particleSpecies[i * particleSpecies_s];
   PartListOffset[i * PartListOffset_s] = specieOffset[ityp-1]+specieCount[ityp-1];
   specieCount[ityp-1]++;
 }

#pragma omp parallel for private(i) schedule(static) if(partCounted > 100)
 for(i=0;i<partCounted;i++){
    long dd,ll,jj,tt,ityp,jtyp;
    real o[3],t[3];
    real dij;
    long numOfPartNeigh=0;
    ityp = (long)particleSpecies[i * particleSpecies_s] - 1;
    real* cip = coords + i * coords_s0;
    real* partNLp = neighListOfParts + i * neighListOfParts_s0;
    real* partNLIp = neighListOfPartsPair + i * neighListOfPartsPair_s0;
    real* Rijpi = Rij_list + i * Rij_list_s0;
    real* RijIpi = Rij_list_Pair + i * Rij_list_Pair_s0;
    real* numNLp = numOfPartNeighListPair + i * numOfPartNeighListPair_s0;
    for(dd=0;dd<3;dd++) o[dd] = cip[dd * coords_s1];
    //for(l=0;l<125;l++){
    for(ll=0;ll<27;ll++){
       for(jj=0;jj<partCounted;jj++){
          real* cjp = coords + jj * coords_s0;
          jtyp = (long)particleSpecies[jj * particleSpecies_s] - 1;
          real* partNLTp = partNLIp + jtyp * neighListOfPartsPair_s1;
          real* RijTpi = RijIpi + jtyp * Rij_list_Pair_s1;
          for(dd=0;dd<3;dd++) t[dd] = scaledVec[ll][dd] + cjp[dd * coords_s1];
#if defined(TH_REAL_IS_FLOAT)
          dij=sqrtf((t[0]-o[0])*(t[0]-o[0])+(t[1]-o[1])*(t[1]-o[1])+(t[2]-o[2])*(t[2]-o[2]));
#else
          dij= sqrt((t[0]-o[0])*(t[0]-o[0])+(t[1]-o[1])*(t[1]-o[1])+(t[2]-o[2])*(t[2]-o[2]));
#endif
          if (dij < cutoff && dij > 0.1) {
             partNLp[numOfPartNeigh * neighListOfParts_s1]=jj;
             real* Rijpj = Rijpi + numOfPartNeigh * Rij_list_s1;
             for(dd=0;dd<3;dd++) Rijpj[dd * Rij_list_s2] = o[dd]-t[dd]; 
             numOfPartNeigh++;
             if (ityp == jtyp) {
                if (dij < pairCutoffs[pairCutoffs_s * jtyp]) {
                   partNLTp[(long)numNLp[jtyp * numOfPartNeighListPair_s1] * neighListOfPartsPair_s2]=jj;
                   real* RijTpj = RijTpi + (long)numNLp[jtyp * numOfPartNeighListPair_s1] * Rij_list_Pair_s2;
                   for(dd=0;dd<3;dd++) RijTpj[dd * Rij_list_Pair_s3] = Rijpj[dd * Rij_list_s2];
                   numNLp[jtyp * numOfPartNeighListPair_s1]++;
                } 
             } else {
                if (dij < pairCutoffs[pairCutoffs_s * (num_typ + ityp + jtyp - 1)]) {
                   partNLTp[(long)numNLp[jtyp * numOfPartNeighListPair_s1] * neighListOfPartsPair_s2]=jj;
                   real* RijTpj = RijTpi + (long)numNLp[jtyp * numOfPartNeighListPair_s1] * Rij_list_Pair_s2;
                   for(dd=0;dd<3;dd++) RijTpj[dd * Rij_list_Pair_s3] = Rijpj[dd * Rij_list_s2];
                   numNLp[jtyp * numOfPartNeighListPair_s1]++;
                } 
             }
          } 
       }
    }
    numOfPartNeighList[i * numOfPartNeighList_s] = numOfPartNeigh;
 }

  free(num_atom_typs);  
  free(specieCount);  
  free(specieOffset);  
  
  lua_getfield(L, 1, "outputList");
  return 1;
}


static int poten_(Energy_neighborUpdateAll)(lua_State *L)
{
  THTensor *coords_ptr = luaT_getfieldcheckudata(L, 1, "atomPositions", torch_Tensor);
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_syms = luaT_getfieldchecknumber(L, 1, "numSyms");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  real cutoff = luaT_getfieldchecknumber(L, 1, "potCutoff");
  real neighbin = luaT_getfieldchecknumber(L, 1, "neighborBin");
  THTensor * currentPartList_ptr = luaT_getfieldcheckudata(L, 1, "partList", torch_Tensor);
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * puv = luaT_getfieldcheckudata(L, 1, "unitCell", torch_Tensor);
  THTensor * numOfPartNeighList_ptr = luaT_getfieldcheckudata(L, 1, "numNeighList", torch_Tensor);
  THTensor * neighListOfParts_ptr = luaT_getfieldcheckudata(L, 1, "partNeighList", torch_Tensor);
  THTensor * Rij_list_ptr = luaT_getfieldcheckudata(L, 1, "RijList", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);

//  real* input_data  = THTensor_(data)(input);
  real * currentPartList = THTensor_(data)(currentPartList_ptr);
  long   currentPartList_s = THTensor_(stride)(currentPartList_ptr, 0);
  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * coords = THTensor_(data)(coords_ptr);
  long   coords_s0 = THTensor_(stride)(coords_ptr, 0);
  long   coords_s1 = THTensor_(stride)(coords_ptr, 1);
  real * uv = THTensor_(data)(puv);
  long   uv_s0 = THTensor_(stride)(puv, 0);
  long   uv_s1 = THTensor_(stride)(puv, 1);
  real * uvp;

  THTensor_(resize1d)(numOfPartNeighList_ptr, partCounted);
  real * numOfPartNeighList  = THTensor_(data)(numOfPartNeighList_ptr);
  long   numOfPartNeighList_s = THTensor_(stride)(numOfPartNeighList_ptr, 0);
  THTensor_(resize2d)(neighListOfParts_ptr, partCounted, 2*partCounted);
  real * neighListOfParts  = THTensor_(data)(neighListOfParts_ptr);
  long   neighListOfParts_s0 = THTensor_(stride)(neighListOfParts_ptr, 0);
  long   neighListOfParts_s1 = THTensor_(stride)(neighListOfParts_ptr, 1);
  
  THTensor_(resize3d)(Rij_list_ptr, partCounted, 2*partCounted, 3);
  real * Rij_list  = THTensor_(data)(Rij_list_ptr);
  long   Rij_list_s0 = THTensor_(stride)(Rij_list_ptr, 0);
  long   Rij_list_s1 = THTensor_(stride)(Rij_list_ptr, 1);
  long   Rij_list_s2 = THTensor_(stride)(Rij_list_ptr, 2);

  long i,j,k,d;
  long l,m,n;
//  long numOfPartNeigh;
  real Rsqij, Rsqik, Rsqjk;
  real cutsq = cutoff * cutoff; 
  if(neighbin > 0.0) cutsq = cutsq + (neighbin * neighbin);
  long offset;

  long* num_atom_typs = (long*) malloc(num_typ*sizeof(long));
  
  for(d = 0; d < num_typ; d++) {
      num_atom_typs[d]=0;
  }
  for (i = 0; i < partCounted; ++i) {
    for(d = 0; d < num_typ; d++) {
       if((long)particleSpecies[i * particleSpecies_s]-1 == d){
          num_atom_typs[d]++;
       }
    }
  }
  num_syms = num_typ * (num_rad + num_ang) + 
             ( poten_(Energy_c_fact)(L, num_typ) / ( 2 * poten_(Energy_c_fact)(L, num_typ-2))) * num_ang;
  
  THTensor_(resize1d)(PartListOffset_ptr, partCounted);
  real* PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long  PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);
    
  long*  specieCount = (long *) malloc(num_typ*sizeof(long));
  long* specieOffset = (long *) malloc((num_typ+1)*sizeof(long));
  
  specieOffset[0]=0;
  for (d=0;d<num_typ;d++){
        specieCount[d]=0;
        if (d<1) { // Second specie's offset
          specieOffset[d+1] = num_atom_typs[d];
        } else { // Shift others accordingly
          specieOffset[d+1] = specieOffset[d] + num_atom_typs[d];
        }
  }

  // Calculate neighbor lists over all atoms
 
 real scaledVec[27][3];
 //real scaledVec[125][3];
 i=0;
 for(l=-1;l<=1;l++){
   for(m=-1;m<=1;m++){
     for(n=-1;n<=1;n++){
         for(d=0;d<3;d++) { 
            uvp = uv + d * uv_s1;
            scaledVec[i][d] = l * uvp[0] + m * uvp[uv_s0] + n * uvp[2 * uv_s0];
         }
         i++;
     }
   }
 }

//    real * cip;
//    real * cjp;
    long  ityp;
//    long dd,ll;
//    real o[3],t[3];
//    real dij;
//    long numOfPartNeigh=0;
//    real *partNLp;
//    real *Rijpi;
//    real *Rijpj;

 for(i=0;i<partCounted;i++){
   ityp =  (long)particleSpecies[i * particleSpecies_s];
   PartListOffset[i * PartListOffset_s] = specieOffset[ityp-1]+specieCount[ityp-1];
   specieCount[ityp-1]++;
 }

#pragma omp parallel for private(i) schedule(static) if(partCounted > 100)
 for(i=0;i<partCounted;i++){
    long dd,ll,jj;
    real o[3],t[3];
    real dij;
    long numOfPartNeigh=0;
    real* cip = coords + i * coords_s0;
    real* partNLp = neighListOfParts + i * neighListOfParts_s0;
    real* Rijpi = Rij_list + i * Rij_list_s0;
    for(dd=0;dd<3;dd++) o[dd] = cip[dd * coords_s1];
    //for(l=0;l<125;l++){
    for(ll=0;ll<27;ll++){
       for(jj=0;jj<partCounted;jj++){
          real* cjp = coords + jj * coords_s0;
          for(dd=0;dd<3;dd++) t[dd] = scaledVec[ll][dd] + cjp[dd * coords_s1];
#if defined(TH_REAL_IS_FLOAT)
          dij=sqrtf((t[0]-o[0])*(t[0]-o[0])+(t[1]-o[1])*(t[1]-o[1])+(t[2]-o[2])*(t[2]-o[2]));
#else
          dij=sqrt((t[0]-o[0])*(t[0]-o[0])+(t[1]-o[1])*(t[1]-o[1])+(t[2]-o[2])*(t[2]-o[2]));
#endif
          if (dij < cutoff && dij > 0.1) {
             partNLp[numOfPartNeigh * neighListOfParts_s1]=jj;
             real* Rijpj = Rijpi + numOfPartNeigh * Rij_list_s1;
             for(dd=0;dd<3;dd++) Rijpj[dd * Rij_list_s2] = o[dd]-t[dd]; 
             numOfPartNeigh++;
          } 
       }
    }
    numOfPartNeighList[i * numOfPartNeighList_s] = numOfPartNeigh;
 }

  free(num_atom_typs);  
  free(specieCount);  
  free(specieOffset);  
  
  lua_getfield(L, 1, "outputList");
  return 1;
}

static int poten_(Energy_inputUpdateAll)(lua_State *L)
{
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_syms = luaT_getfieldchecknumber(L, 1, "numSyms");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  THTensor * rad_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaRad", torch_Tensor);
  THTensor * ang_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaAng", torch_Tensor);
  THTensor * lambda_ptr = luaT_getfieldcheckudata(L, 1, "lambdaAng", torch_Tensor);
  THTensor * zeta_ptr = luaT_getfieldcheckudata(L, 1, "zetaAng", torch_Tensor);
  real cutoff = luaT_getfieldchecknumber(L, 1, "potCutoff");
  real neighbin = luaT_getfieldchecknumber(L, 1, "neighborBin");
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * numOfPartNeighList_ptr = luaT_getfieldcheckudata(L, 1, "numNeighList", torch_Tensor);
  THTensor * neighListOfParts_ptr = luaT_getfieldcheckudata(L, 1, "partNeighList", torch_Tensor);
  THTensor * Rij_list_ptr = luaT_getfieldcheckudata(L, 1, "RijList", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  THTensor_(resize2d)(output, partCounted, num_syms);
  THTensor_(zero)(output);
  real* output_data = THTensor_(data)(output);
  long  output_s0 = THTensor_(stride)(output, 0);
  long  output_s1 = THTensor_(stride)(output, 1);
  real* outputp;

//  real* input_data  = THTensor_(data)(input);
  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long   PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);
  
  real* rad_eta  = THTensor_(data)(rad_eta_ptr);
  long  rad_eta_s = THTensor_(stride)(rad_eta_ptr, 0);
  real* ang_eta  = THTensor_(data)(ang_eta_ptr);
  long  ang_eta_s = THTensor_(stride)(ang_eta_ptr, 0);
  real* lambda  = THTensor_(data)(lambda_ptr);
  long  lambda_s = THTensor_(stride)(lambda_ptr, 0);
  real* zeta  = THTensor_(data)(zeta_ptr);
  long  zeta_s = THTensor_(stride)(zeta_ptr, 0);

  real* numOfPartNeighList  = THTensor_(data)(numOfPartNeighList_ptr);
  long  numOfPartNeighList_s = THTensor_(stride)(numOfPartNeighList_ptr, 0);
  real* neighListOfParts  = THTensor_(data)(neighListOfParts_ptr);
  long  neighListOfParts_s0 = THTensor_(stride)(neighListOfParts_ptr, 0);
  long  neighListOfParts_s1 = THTensor_(stride)(neighListOfParts_ptr, 1);
  real* partNLp;
  real* Rij_list  = THTensor_(data)(Rij_list_ptr);
  long  Rij_list_s0 = THTensor_(stride)(Rij_list_ptr, 0);
  long  Rij_list_s1 = THTensor_(stride)(Rij_list_ptr, 1);
  long  Rij_list_s2 = THTensor_(stride)(Rij_list_ptr, 2);
  real* Rijpi;
  real* Rijpj;
  real* Rijpk;
  

  long i,j,k,jj,kk,d;
  long l,m,n,symi;
  long numOfPartNeigh;
  real Rsqij, Rsqik, Rsqjk;
  real cutsq = cutoff * cutoff; 
  if(neighbin > 0.0) cutsq = cutsq + (neighbin * neighbin);
  real Rij[3],Rik[3],Rjk[3];
  real fcutall, fcutik, fcutij, fcutjk;
  real costhetaVALUE, costheta;
  real power1, power2;
  long offset;
  real pi=3.1415926535897932384626433832795028841971694;
  real dij,djk,dik,dall;
  long ityp;
  long jtyp;
  long ktyp;

 for(i=0;i<partCounted;i++){
    numOfPartNeigh = (long)numOfPartNeighList[i * numOfPartNeighList_s];
    partNLp = neighListOfParts + i * neighListOfParts_s0;
    ityp =  (long)particleSpecies[i * particleSpecies_s];
    outputp = output_data + (long)PartListOffset[i * PartListOffset_s] * output_s0;
    Rijpi = Rij_list + i * Rij_list_s0;
    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = (long)partNLp[jj * neighListOfParts_s1];
      Rsqij = 0.0;
      Rijpj = Rijpi + jj * Rij_list_s1;
      for (d = 0; d < 3; ++d) {
        Rij[d] =  Rijpj[d * Rij_list_s2]; 
        Rsqij +=  Rij[d] * Rij[d]; 
      } /* End of Rij[k] loop */

      if (Rsqij < cutsq && sqrt(Rsqij > 0.1) ) {
      //if (Rsqij < cutsq && Rsqij > 0.000000001) {
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        offset = (jtyp-1) * num_rad;

#if defined(TH_REAL_IS_FLOAT)
        dij = sqrtf(Rsqij);
        fcutij = (dij < cutoff) ? 0.5 * (cosf(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            outputp[(symi+offset) * output_s1] +=  (expf(-rad_eta[rad_eta_s * symi] * dij * dij) * fcutij ); 
        }
#else
        dij =  sqrt(Rsqij);
        fcutij = (dij < cutoff) ? 0.5 * ( cos(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            outputp[(symi+offset) * output_s1] +=  ( exp(-rad_eta[rad_eta_s * symi] * dij * dij) * fcutij ); 
        }
#endif

        for (kk = 0; kk < numOfPartNeigh; ++kk) {
          if (jj==kk) continue;
          k = (long)partNLp[kk * neighListOfParts_s1];
          //if (j==k && i==k) continue;
          
          Rsqik = 0.0;
          Rsqjk = 0.0;
          Rijpk = Rijpi + kk * Rij_list_s1;
          for (d = 0; d < 3; ++d) {
            Rik[d] = Rijpk[d * Rij_list_s2]; 
            Rjk[d] = Rik[d] - Rij[d]; 
            Rsqik += Rik[d] * Rik[d]; 
            Rsqjk += Rjk[d] * Rjk[d]; 
          } /* End of Rij[k] loop */
        
          //if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.000000001 && Rsqjk > 0.000000001) {
          if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.1 && Rsqjk > 0.1) {
            costheta = 0.0;
            for (d = 0; d < 3; ++d) costheta += Rij[d]*Rik[d];
            ktyp =  (long)particleSpecies[k * particleSpecies_s];
            offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);

#if defined(TH_REAL_IS_FLOAT)
            djk = sqrtf(Rsqjk);
            dik = sqrtf(Rsqik);
            fcutjk = (djk < cutoff) ? 0.5 * (cosf(pi * djk / cutoff ) + 1.0) : 0.0 ; 
            fcutik = (dik < cutoff) ? 0.5 * (cosf(pi * dik / cutoff ) + 1.0) : 0.0 ; 
#else
            djk =  sqrt(Rsqjk);
            dik =  sqrt(Rsqik);
            fcutjk = (djk < cutoff) ? 0.5 * ( cos(pi * djk / cutoff ) + 1.0) : 0.0 ; 
            fcutik = (dik < cutoff) ? 0.5 * ( cos(pi * dik / cutoff ) + 1.0) : 0.0 ; 
#endif

            fcutall = fcutik * fcutij * fcutjk;
            dall = dik*dik + dij*dij + djk*djk ;
            costhetaVALUE = costheta / ( dij * dik);

#if defined(TH_REAL_IS_FLOAT)
            for (symi=0;symi<num_ang;symi++) { 
                power1 = powf(1.0 + lambda[lambda_s * symi] * costhetaVALUE, zeta[zeta_s * symi]);
                power2 = powf(2.0,1.0 - zeta[zeta_s * symi]);
                outputp[(symi+offset) * output_s1] += ( expf( -ang_eta[ang_eta_s * symi] * dall) * fcutall * power1 * power2 );
            }
#else
            for (symi=0;symi<num_ang;symi++) { 
                power1 = pow(1.0 + lambda[lambda_s * symi] * costhetaVALUE, zeta[zeta_s * symi]);
                power2 = pow(2.0,1.0 - zeta[zeta_s * symi]);
                outputp[(symi+offset) * output_s1] +=  ( exp( -ang_eta[ang_eta_s * symi] * dall) * fcutall * power1 * power2 );
            }
#endif

          } /* End of if inside cutoff, Rsqik < cutoff */
        }  /* loop on kk */
      } /* End of if inside cutoff, Rsqij < cutoff */
    }  /* loop on jj */
  }  /* infinite while loop on i (terminated by break statements above) */

  lua_getfield(L, 1, "output");
  return 1;
}

static int poten_(Energy_updateAll)(lua_State *L)
{
  THTensor *coords_ptr = luaT_getfieldcheckudata(L, 1, "atomPositions", torch_Tensor);
//  THTensor *inputList = luaT_checkudata(L, 3, torch_Tensor);
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  THTensor * rad_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaRad", torch_Tensor);
  THTensor * ang_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaAng", torch_Tensor);
  THTensor * lambda_ptr = luaT_getfieldcheckudata(L, 1, "lambdaAng", torch_Tensor);
  THTensor * zeta_ptr = luaT_getfieldcheckudata(L, 1, "zetaAng", torch_Tensor);
  real cutoff = luaT_getfieldchecknumber(L, 1, "potCutoff");
  real neighbin = luaT_getfieldchecknumber(L, 1, "neighborBin");
  THTensor * currentPartList_ptr = luaT_getfieldcheckudata(L, 1, "partList", torch_Tensor);
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * puv = luaT_getfieldcheckudata(L, 1, "unitCell", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

//  real* input_data  = THTensor_(data)(input);
  real * currentPartList = THTensor_(data)(currentPartList_ptr);
  long   currentPartList_s = THTensor_(stride)(currentPartList_ptr, 0);
  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * coords = THTensor_(data)(coords_ptr);
  long   coords_s0 = THTensor_(stride)(coords_ptr, 0);
  long   coords_s1 = THTensor_(stride)(coords_ptr, 1);
  real * uv = THTensor_(data)(puv);
  long   uv_s0 = THTensor_(stride)(puv, 0);
  long   uv_s1 = THTensor_(stride)(puv, 1);
  real*  uvp;

  real* rad_eta  = THTensor_(data)(rad_eta_ptr);
  long  rad_eta_s = THTensor_(stride)(rad_eta_ptr, 0);
  real* ang_eta  = THTensor_(data)(ang_eta_ptr);
  long  ang_eta_s = THTensor_(stride)(ang_eta_ptr, 0);
  real* lambda  = THTensor_(data)(lambda_ptr);
  long  lambda_s = THTensor_(stride)(lambda_ptr, 0);
  real* zeta  = THTensor_(data)(zeta_ptr);
  long  zeta_s = THTensor_(stride)(zeta_ptr, 0);

  long i,j,k,jj,kk,d;
  long l,m,n,symi;
  long numOfPartNeigh;
  long neighCount;
  real Rsqij, Rsqik, Rsqjk;
  real cutsq = cutoff * cutoff; 
  if(neighbin > 0.0) cutsq = cutsq + (neighbin * neighbin);
  real Rij[3],Rik[3],Rjk[3];
  real fcutall, fcutik, fcutij, fcutjk;
  real costhetaVALUE, costheta;
  real power1, power2;
  long offset, num_syms;
  real pi=3.1415926535897932384626433832795028841971694;

  long* num_atom_typs = (long*) malloc(num_typ*sizeof(long));
  
  for(d = 0; d < num_typ; d++) {
      num_atom_typs[d]=0;
  }
  for (i = 0; i < partCounted; ++i) {
    for(d = 0; d < num_typ; d++) {
       if((long)particleSpecies[i * particleSpecies_s]-1 == d){
          num_atom_typs[d]++;
       }
    }
  }
  num_syms = num_typ * (num_rad + num_ang) + 
             ( poten_(Energy_c_fact)(L, num_typ) / ( 2 * poten_(Energy_c_fact)(L, num_typ-2))) * num_ang;
  
  THTensor_(resize2d)(output, partCounted, num_syms);
  THTensor_(zero)(output);
  real* output_data = THTensor_(data)(output);
  long  output_s0 = THTensor_(stride)(output, 0);
  long  output_s1 = THTensor_(stride)(output, 1);
  real* outputp;
  
  THTensor_(resize1d)(PartListOffset_ptr, partCounted);
  real* PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long  PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);
    
  long*  specieCount = (long *) malloc(num_typ*sizeof(long));
  long* specieOffset = (long *) malloc((num_typ+1)*sizeof(long));
  
  specieOffset[0]=0;
  for (d=0;d<num_typ;d++){
        specieCount[d]=0;
        if (d<1) { // First specie's offset
           if (num_atom_typs[d]>0) {
               specieOffset[d+1] = num_atom_typs[d];
           } else {
               specieOffset[d+1] = 0;
           }
        } else { // Shift others accordingly
           if (num_atom_typs[d]>0) {
               specieOffset[d+1] = specieOffset[d] + num_atom_typs[d];
           } else {
               specieOffset[d+1] = specieOffset[d];
           }
        }
  }
  long* neighListOfParts = (long *) malloc(2*partCounted*partCounted*sizeof(long)); 
  real* Rij_list = (real *) malloc(2*partCounted*partCounted*3*sizeof(real)); 
  long* numOfPartNeighList = (long *) malloc(partCounted*sizeof(long)); 
  long* neighListOffset = (long *) malloc(partCounted*sizeof(long)); 
  //long* PartListOffset = (long *) malloc(partCounted*sizeof(long)); 


  // Calculate neighbor lists over all atoms
 real o[3],t[3];
 //THTensor * o = THTensor_(newWithSize1d)((long)3);
 //THTensor * t = THTensor_(newWithSize1d)((long)3);
 
 real scaledVec[125][3];
 real dij,djk,dik,dall;
 i=0;
 for(l=-2;l<=2;l++){
   for(m=-2;m<=2;m++){
     for(n=-2;n<=2;n++){
         for(d=0;d<3;d++) { 
            uvp = uv + d * uv_s1;
            scaledVec[i][d] = l * uvp[0] + m * uvp[uv_s0] + n * uvp[2 * uv_s0];
         }
         i++;
     }
   }
 }
 real * cip;
 real * cjp;
 long  ityp;
 long  jtyp;
 long  ktyp;
 neighCount=0;
 for(i=0;i<partCounted;i++){
    numOfPartNeigh=0;
    neighListOffset[i]=neighCount;
    cip = coords + i * coords_s0;
    for(d=0;d<3;d++) o[d] = cip[d * coords_s1];
    for(l=0;l<125;l++){
       for(j=0;j<partCounted;j++){
          cjp = coords + j * coords_s0;
          for(d=0;d<3;d++) t[d] = scaledVec[l][d] + cjp[d * coords_s1];
          dij=(t[0]-o[0])*(t[0]-o[0])+(t[1]-o[1])*(t[1]-o[1])+(t[2]-o[2])*(t[2]-o[2]);
          if (dij < cutsq && dij > 0.01) {
             neighListOfParts[neighCount]=j;
             for(d=0;d<3;d++) Rij_list[(neighCount*3)+d]=t[d]-o[d]; 
             neighCount++; 
             numOfPartNeigh++;
          } 
       }
    }
    ityp =  (long)particleSpecies[i * particleSpecies_s];
    PartListOffset[i * PartListOffset_s] = specieOffset[ityp-1]+specieCount[ityp-1];
    specieCount[ityp-1]++;
    numOfPartNeighList[i]=numOfPartNeigh;
 }


 for(i=0;i<partCounted;i++){
    numOfPartNeigh = numOfPartNeighList[i];
    neighCount = neighListOffset[i];
    ityp =  (long)particleSpecies[i * particleSpecies_s];
    outputp = output_data + (long)PartListOffset[i * PartListOffset_s] * output_s0;
    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = neighListOfParts[neighCount+jj];
      Rsqij = 0.0;
      for (d = 0; d < 3; ++d) {
        Rij[d] =  Rij_list[(neighCount+jj)*3 + d]; 
        Rsqij +=  Rij[d] * Rij[d]; 
      } /* End of Rij[k] loop */

      if (Rsqij < cutsq) {
      //if (Rsqij < cutsq && Rsqij > 0.000000001) {
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        offset = (jtyp-1) * num_rad;

#if defined(TH_REAL_IS_FLOAT)
        dij = sqrtf(Rsqij);
        fcutij = (dij < cutoff) ? 0.5 * (cosf(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            outputp[(symi+offset) * output_s1] +=  (expf(-rad_eta[rad_eta_s * symi] * dij * dij) * fcutij ); 
        }
#else
        dij =  sqrt(Rsqij);
        fcutij = (dij < cutoff) ? 0.5 * ( cos(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            outputp[(symi+offset) * output_s1] +=  ( exp(-rad_eta[rad_eta_s * symi] * dij * dij) * fcutij ); 
        }
#endif

        for (kk = 0; kk < numOfPartNeigh; ++kk) {
          if (jj==kk) continue;
          k = neighListOfParts[neighCount+kk];
          //if (j==k && i==k) continue;
          
          Rsqik = 0.0;
          Rsqjk = 0.0;
          for (d = 0; d < 3; ++d) {
            Rik[d] = Rij_list[(neighCount+kk)*3 + d]; 
            Rjk[d] = Rik[d] - Rij[d]; 
            Rsqik += Rik[d] * Rik[d]; 
            Rsqjk += Rjk[d] * Rjk[d]; 
          } /* End of Rij[k] loop */
        
          //if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.000000001 && Rsqjk > 0.000000001) {
          if (Rsqik < cutsq && Rsqik > 0.001 && Rsqjk > 0.001) {
            costheta = 0.0;
            for (d = 0; d < 3; ++d) costheta += Rij[d]*Rik[d];
            ktyp =  (long)particleSpecies[k * particleSpecies_s];
            offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);

#if defined(TH_REAL_IS_FLOAT)
            djk = sqrtf(Rsqjk);
            dik = sqrtf(Rsqik);
            fcutjk = (djk < cutoff) ? 0.5 * (cosf(pi * djk / cutoff ) + 1.0) : 0.0 ; 
            fcutik = (dik < cutoff) ? 0.5 * (cosf(pi * dik / cutoff ) + 1.0) : 0.0 ; 
#else
            djk =  sqrt(Rsqjk);
            dik =  sqrt(Rsqik);
            fcutjk = (djk < cutoff) ? 0.5 * ( cos(pi * djk / cutoff ) + 1.0) : 0.0 ; 
            fcutik = (dik < cutoff) ? 0.5 * ( cos(pi * dik / cutoff ) + 1.0) : 0.0 ; 
#endif

            fcutall = fcutik * fcutij * fcutjk;
            dall = dik*dik + dij*dij + djk*djk ;
            costhetaVALUE = costheta / ( dij * dik);

#if defined(TH_REAL_IS_FLOAT)
            for (symi=0;symi<num_ang;symi++) { 
                power1 = powf(1.0 + lambda[lambda_s * symi] * costhetaVALUE, zeta[zeta_s * symi]);
                power2 = powf(2.0,1.0 - zeta[zeta_s * symi]);
                outputp[(symi+offset) * output_s1] += ( expf( -ang_eta[ang_eta_s * symi] * dall) * fcutall * power1 * power2 );
            }
#else
            for (symi=0;symi<num_ang;symi++) { 
                power1 = pow(1.0 + lambda[lambda_s * symi] * costhetaVALUE, zeta[zeta_s * symi]);
                power2 = pow(2.0,1.0 - zeta[zeta_s * symi]);
                outputp[(symi+offset) * output_s1] +=  ( exp( -ang_eta[ang_eta_s * symi] * dall) * fcutall * power1 * power2 );
            }
#endif

          } /* End of if inside cutoff, Rsqik < cutoff */
        }  /* loop on kk */
      } /* End of if inside cutoff, Rsqij < cutoff */
    }  /* loop on jj */
  }  /* infinite while loop on i (terminated by break statements above) */

  free(num_atom_typs);  
  free(specieCount);  
  free(specieOffset);  
  free(neighListOfParts);
  free(Rij_list); 
  free(numOfPartNeighList);
  free(neighListOffset);
  
  lua_getfield(L, 1, "output");
  return 1;
}

static int poten_(Energy_updateAllForces)(lua_State *L)
{
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_syms = luaT_getfieldchecknumber(L, 1, "numSyms");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  THTensor * rad_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaRad", torch_Tensor);
  THTensor * ang_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaAng", torch_Tensor);
  THTensor * lambda_ptr = luaT_getfieldcheckudata(L, 1, "lambdaAng", torch_Tensor);
  THTensor * zeta_ptr = luaT_getfieldcheckudata(L, 1, "zetaAng", torch_Tensor);
  real cutoff = luaT_getfieldchecknumber(L, 1, "potCutoff");
  real neighbin = luaT_getfieldchecknumber(L, 1, "neighborBin");
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * numOfPartNeighList_ptr = luaT_getfieldcheckudata(L, 1, "numNeighList", torch_Tensor);
  THTensor * neighListOfParts_ptr = luaT_getfieldcheckudata(L, 1, "partNeighList", torch_Tensor);
  THTensor * Rij_list_ptr = luaT_getfieldcheckudata(L, 1, "RijList", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  THTensor_(resize2d)(output, partCounted, 3);
  THTensor_(zero)(output);
  real* output_data = THTensor_(data)(output);
  long  output_s0 = THTensor_(stride)(output, 0);
  long  output_s1 = THTensor_(stride)(output, 1);
  real* outputpi;
  real* outputpj;
  real* outputpk;

  real* input_data  = THTensor_(data)(input);
  long  input_s0 = THTensor_(stride)(input, 0);
  long  input_s1 = THTensor_(stride)(input, 1);
  real* inputp;

  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long   PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);

  real* rad_eta  = THTensor_(data)(rad_eta_ptr);
  long  rad_eta_s = THTensor_(stride)(rad_eta_ptr, 0);
  real* ang_eta  = THTensor_(data)(ang_eta_ptr);
  long  ang_eta_s = THTensor_(stride)(ang_eta_ptr, 0);
  real* lambda  = THTensor_(data)(lambda_ptr);
  long  lambda_s = THTensor_(stride)(lambda_ptr, 0);
  real* zeta  = THTensor_(data)(zeta_ptr);
  long  zeta_s = THTensor_(stride)(zeta_ptr, 0);

  real* numOfPartNeighList  = THTensor_(data)(numOfPartNeighList_ptr);
  long  numOfPartNeighList_s = THTensor_(stride)(numOfPartNeighList_ptr, 0);
  real* neighListOfParts  = THTensor_(data)(neighListOfParts_ptr);
  long  neighListOfParts_s0 = THTensor_(stride)(neighListOfParts_ptr, 0);
  long  neighListOfParts_s1 = THTensor_(stride)(neighListOfParts_ptr, 1);
  real* partNLp;
  real* Rij_list  = THTensor_(data)(Rij_list_ptr);
  long  Rij_list_s0 = THTensor_(stride)(Rij_list_ptr, 0);
  long  Rij_list_s1 = THTensor_(stride)(Rij_list_ptr, 1);
  long  Rij_list_s2 = THTensor_(stride)(Rij_list_ptr, 2);
  real* Rijpi;
  real* Rijpj;
  real* Rijpk;
  
  long i,j,k,jj,kk,d;
  long numOfPartNeigh;
  real Rsqij,Rsqik,Rsqjk;
  real cutsq = cutoff * cutoff; 
  if(neighbin > 0.0) cutsq = cutsq + (neighbin * neighbin);
  real Rij[3],Rik[3],Rjk[3];
  real fcutall, fcutik, fcutij, fcutjk;
  real fcut_gradi_ij, fcut_gradi_ik, fcut_gradj_jk;
  real thetagradi[3],thetagradj[3],thetagradk[3];
  real costheta, costhetaVALUE;
  real power1, power2, vexp1, vexp2;
  real powksi1, expon, angexp;
  real RijRik, dijdijV, dijdij, dijdikV, dikdikV;
  real ffi[3],ffj[3],ffk[3];
  long offset, symi;
  real pi=3.1415926535897932384626433832795028841971694;
  real dij,djk,dik,dall;
  long ityp;
  long jtyp;
  long ktyp;

 for(i=0;i<partCounted;i++){
    numOfPartNeigh = (long)numOfPartNeighList[i * numOfPartNeighList_s];
    partNLp = neighListOfParts + i * neighListOfParts_s0;
    ityp = (long)particleSpecies[i * particleSpecies_s];
    outputpi = output_data + i * output_s0;
    inputp = input_data + (long)PartListOffset[i * PartListOffset_s] * input_s0;
    Rijpi = Rij_list + i * Rij_list_s0;

    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = (long)partNLp[jj * neighListOfParts_s1];
      Rsqij = 0.0;
      Rijpj = Rijpi + jj * Rij_list_s1;
      for (d = 0; d < 3; ++d) {
        Rij[d] =  Rijpj[d * Rij_list_s2]; 
        Rsqij +=  Rij[d] * Rij[d]; 
      } /* End of Rij[k] loop */

      if (Rsqij < cutsq && Rsqij > 0.000000001) {
        for (d = 0; d < 3; d++)  ffi[d] = 0.0;
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        offset = (jtyp-1) * num_rad;
        outputpj = output_data + j * output_s0;

#if defined(TH_REAL_IS_FLOAT)
        dij = sqrtf(Rsqij);
        dijdij = dij * dij;
        fcutij = (dij < cutoff) ? 0.5 * (cosf(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        fcut_gradi_ij = (dij < cutoff) ? -(0.5 * pi/cutoff ) * sinf(pi * dij/cutoff ) / dij : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            vexp2 = inputp[(offset + symi) * input_s1] * expf( -rad_eta[rad_eta_s * symi] * dijdij); 
            vexp1 = -2.0 * rad_eta[rad_eta_s * symi] * fcutij * vexp2; 
            for (d = 0; d < 3; d++) ffi[d] += ( vexp1 + vexp2 * fcut_gradi_ij ) * Rij[d]; 
        }
#else
        dij = sqrt(Rsqij);
        dijdij = dij * dij;
        fcutij = (dij < cutoff) ? 0.5 * (cos(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        fcut_gradi_ij = (dij < cutoff) ? -(0.5 * pi/cutoff ) * sin(pi * dij/cutoff ) / dij : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            vexp2 = inputp[(offset + symi) * input_s1] * exp( -rad_eta[rad_eta_s * symi] * dijdij); 
            vexp1 = -2.0 * rad_eta[rad_eta_s * symi] * fcutij * vexp2; 
            for (d = 0; d < 3; d++) ffi[d] += ( vexp1 + vexp2 * fcut_gradi_ij ) * Rij[d]; 
        }
#endif
        for (d = 0; d < 3; d++) {  outputpi[d * output_s1] -= ffi[d]; 
                                   outputpj[d * output_s1] += ffi[d]; }

        for (kk = 0; kk < numOfPartNeigh; ++kk) {
           if (jj==kk) continue;
           k = (long)partNLp[kk * neighListOfParts_s1];
           if (j==k && i==k) continue;

          Rsqik = 0.0;
          Rsqjk = 0.0;
          Rijpk = Rijpi + kk * Rij_list_s1;
          for (d = 0; d < 3; ++d) {
            Rik[d] = Rijpk[d * Rij_list_s2]; 
            Rjk[d] = Rik[d] - Rij[d]; 
            Rsqik += Rik[d]*Rik[d]; 
            Rsqjk += Rjk[d]*Rjk[d]; 
          } /* End of Rij[k] loop */
        
          if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.000000001 && Rsqjk > 0.000000001) {
          //if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.00001 && Rsqjk > 0.00001) {
             outputpk = output_data + k * output_s0;
             ktyp =  (long)particleSpecies[k * particleSpecies_s];
             offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);
             RijRik = 0.0;
             for (d = 0; d < 3; d++) {   ffi[d] = 0.0; ffj[d] = 0.0; ffk[d] = 0.0; RijRik += Rij[d]*Rik[d]; }

#if defined(TH_REAL_IS_FLOAT)
             dik = sqrtf(Rsqik);
             djk = sqrtf(Rsqjk);
             fcutjk = (djk < cutoff) ? 0.5 * (cosf(pi * djk / cutoff ) + 1.0) : 0.0 ; 
             fcutik = (dik < cutoff) ? 0.5 * (cosf(pi * dik / cutoff ) + 1.0) : 0.0 ; 
             fcut_gradi_ik = (dik < cutoff) ? -(0.5 * pi/cutoff ) * sinf(pi * dik/cutoff ) / dik : 0.0 ; 
             fcut_gradj_jk = (djk < cutoff) ? -(0.5 * pi/cutoff ) * sinf(pi * djk/cutoff ) / djk : 0.0 ; 
#else
             dik = sqrt(Rsqik);
             djk = sqrt(Rsqjk);
             fcutjk = (djk < cutoff) ? 0.5 * (cos(pi * djk / cutoff ) + 1.0) : 0.0 ; 
             fcutik = (dik < cutoff) ? 0.5 * (cos(pi * dik / cutoff ) + 1.0) : 0.0 ; 
             fcut_gradi_ik = (dik < cutoff) ? -(0.5 * pi/cutoff ) * sin(pi * dik/cutoff ) / dik : 0.0 ; 
             fcut_gradj_jk = (djk < cutoff) ? -(0.5 * pi/cutoff ) * sin(pi * djk/cutoff ) / djk : 0.0 ; 
#endif

             dijdijV = 1.0/(dij * dij);
             dikdikV = 1.0/(dik * dik);
             dijdikV = 1.0/(dij * dik);
             for(d=0;d<3;d++){
                thetagradi[d] = ( ( Rij[d] + Rik[d] ) - RijRik * ( Rij[d] * dijdijV + Rik[d] * dikdikV ) ) * dijdikV;
                thetagradj[d] = ( RijRik * Rij[d] * dijdijV - Rik[d] ) * dijdikV;
                thetagradk[d] = ( RijRik * Rik[d] * dikdikV - Rij[d] ) * dijdikV;
             } 

             fcutall = fcutik * fcutij * fcutjk;
             dall = dik*dik + dijdij + djk*djk ;
             costhetaVALUE = RijRik * dijdikV;

             for (symi=0;symi<num_ang;symi++) { 
                 costheta = 1.0 + lambda[lambda_s * symi] * costhetaVALUE;
#if defined(TH_REAL_IS_FLOAT)
                 expon   = expf( -ang_eta[ang_eta_s * symi] * dall);
                 power1  = powf(costheta, zeta[zeta_s * symi]);
                 power2  = inputp[(symi+offset) * input_s1] * powf(2.0,1.0 - zeta[zeta_s * symi]);
                 powksi1 = powf(costheta,(zeta[zeta_s * symi]-1.0)) * zeta[zeta_s * symi] * lambda[lambda_s * symi] * expon * fcutall;
#else
                 expon   = exp( -ang_eta[ang_eta_s * symi] * dall);
                 power1  = pow(costheta, zeta[zeta_s * symi]);
                 power2  = inputp[(symi+offset) * input_s1] * pow(2.0, 1.0 - zeta[zeta_s * symi]);
                 powksi1 = pow(costheta,(zeta[zeta_s * symi]-1.0))  * zeta[zeta_s * symi] * lambda[lambda_s * symi] * expon * fcutall;
#endif
                 angexp = -ang_eta[ang_eta_s * symi] * 2.0 * expon * fcutall;
                 for(d=0;d<3;d++){
                    ffi[d] -=  power2 * ( ( thetagradi[d] * powksi1 ) +  
                               power1 * ( ( (  Rij[d] + Rik[d] ) * angexp ) + 
                                          ( expon * fcutjk * (  fcut_gradi_ij * Rij[d] * fcutik + fcutij *  fcut_gradi_ik * Rik[d] ) )
                                        )
                               );
                    ffj[d] -=  power2 * ( ( thetagradj[d] * powksi1 ) +  
                               power1 * ( ( ( -Rij[d] + Rjk[d] ) * angexp ) + 
                                          ( expon * fcutik * ( -fcut_gradi_ij * Rij[d] * fcutjk + fcutij *  fcut_gradj_jk * Rjk[d] ) )
                                        )
                               );
                    ffk[d] -=  power2 * ( ( thetagradk[d] * powksi1 ) +  
                               power1 * ( ( ( -Rik[d] - Rjk[d] ) * angexp ) + 
                                          ( expon * fcutij * ( -fcut_gradi_ik * Rik[d] * fcutjk - fcutik *  fcut_gradj_jk * Rjk[d] ) )
                                        )
                               );
                 }
             }

             for (d = 0; d < 3; d++) {   outputpi[output_s1 * d] += ffi[d];
                                         outputpj[output_s1 * d] += ffj[d]; 
                                         outputpk[output_s1 * d] += ffk[d]; }
          } /* End of if inside cutoff, Rsqik < cutoff */
        }  /* loop on kk */
      } /* End of if inside cutoff, Rsqij < cutoff */
    }  /* loop on jj */
  }  /* infinite while loop on i (terminated by break statements above) */

  lua_getfield(L, 1, "output");
  return 1;
}


static int poten_(Energy_updateAllForceContributions)(lua_State *L)
{
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_syms = luaT_getfieldchecknumber(L, 1, "numSyms");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  THTensor * rad_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaRad", torch_Tensor);
  THTensor * ang_eta_ptr = luaT_getfieldcheckudata(L, 1, "etaAng", torch_Tensor);
  THTensor * lambda_ptr = luaT_getfieldcheckudata(L, 1, "lambdaAng", torch_Tensor);
  THTensor * zeta_ptr = luaT_getfieldcheckudata(L, 1, "zetaAng", torch_Tensor);
  real cutoff = luaT_getfieldchecknumber(L, 1, "potCutoff");
  real neighbin = luaT_getfieldchecknumber(L, 1, "neighborBin");
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * numOfPartNeighList_ptr = luaT_getfieldcheckudata(L, 1, "numNeighList", torch_Tensor);
  THTensor * neighListOfParts_ptr = luaT_getfieldcheckudata(L, 1, "partNeighList", torch_Tensor);
  THTensor * Rij_list_ptr = luaT_getfieldcheckudata(L, 1, "RijList", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);
  THTensor * Radoutputi_ptr = luaT_getfieldcheckudata(L, 1, "RadOutputi", torch_Tensor);
  THTensor * Angoutputi_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputi", torch_Tensor);
  THTensor * Angoutputj_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputj", torch_Tensor);
  THTensor * Angoutputk_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputk", torch_Tensor);
  THTensor * rlistj_ptr = luaT_getfieldcheckudata(L, 1, "RadListj", torch_Tensor);
  THTensor * alistj_ptr = luaT_getfieldcheckudata(L, 1, "AngListj", torch_Tensor);
  THTensor * alistk_ptr = luaT_getfieldcheckudata(L, 1, "AngListk", torch_Tensor);
  THTensor * radlistnum_ptr = luaT_getfieldcheckudata(L, 1, "RadListNum", torch_Tensor);
  THTensor * anglistnum_ptr = luaT_getfieldcheckudata(L, 1, "AngListNum", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  THTensor_(resize1d)(radlistnum_ptr, partCounted);
  THTensor_(zero)(radlistnum_ptr);
  THTensor_(resize1d)(anglistnum_ptr, partCounted);
  THTensor_(zero)(anglistnum_ptr);
  real* radlistnum = THTensor_(data)(radlistnum_ptr);
  long  radlistnum_s = THTensor_(stride)(radlistnum_ptr, 0);
  real* anglistnum = THTensor_(data)(anglistnum_ptr);
  long  anglistnum_s = THTensor_(stride)(anglistnum_ptr, 0);
  
  THTensor_(resize2d)(output, partCounted, 3);
  THTensor_(zero)(output);
  real* output_data = THTensor_(data)(output);
  long  output_s0 = THTensor_(stride)(output, 0);
  long  output_s1 = THTensor_(stride)(output, 1);
  real* outputpi;
  real* outputpj;
  real* outputpk;
  
  THTensor_(resize4d)(Radoutputi_ptr, partCounted, (long)(partCounted/2) * partCounted, num_rad, 3);
  real * Radoutputi  = THTensor_(data)(Radoutputi_ptr);
  long   Radoutputi_s0 = THTensor_(stride)(Radoutputi_ptr, 0);
  long   Radoutputi_s1 = THTensor_(stride)(Radoutputi_ptr, 1);
  long   Radoutputi_s2 = THTensor_(stride)(Radoutputi_ptr, 2);
  long   Radoutputi_s3 = THTensor_(stride)(Radoutputi_ptr, 3);
  real*  Radoutputip;
  real*  Radoutputipj;
  real*  Radoutputips;
  THTensor_(resize2d)(rlistj_ptr, partCounted, (long)(partCounted/2) * partCounted);
  real * rlistj  = THTensor_(data)(rlistj_ptr);
  long   rlistj_s0 = THTensor_(stride)(rlistj_ptr, 0);
  long   rlistj_s1 = THTensor_(stride)(rlistj_ptr, 1);
  real*  rlistjp;
  
  THTensor_(resize4d)(Angoutputi_ptr, partCounted, (long)(partCounted/2) * partCounted, num_ang, 3);
  real * Angoutputi  = THTensor_(data)(Angoutputi_ptr);
  long   Angoutputi_s0 = THTensor_(stride)(Angoutputi_ptr, 0);
  long   Angoutputi_s1 = THTensor_(stride)(Angoutputi_ptr, 1);
  long   Angoutputi_s2 = THTensor_(stride)(Angoutputi_ptr, 2);
  long   Angoutputi_s3 = THTensor_(stride)(Angoutputi_ptr, 3);
  real*  Angoutputip;
  real*  Angoutputipj;
  real*  Angoutputips;
  
  THTensor_(resize4d)(Angoutputj_ptr, partCounted, (long)(partCounted/2) * partCounted, num_ang, 3);
  real * Angoutputj  = THTensor_(data)(Angoutputj_ptr);
  long   Angoutputj_s0 = THTensor_(stride)(Angoutputj_ptr, 0);
  long   Angoutputj_s1 = THTensor_(stride)(Angoutputj_ptr, 1);
  long   Angoutputj_s2 = THTensor_(stride)(Angoutputj_ptr, 2);
  long   Angoutputj_s3 = THTensor_(stride)(Angoutputj_ptr, 3);
  real*  Angoutputjp;
  real*  Angoutputjpj;
  real*  Angoutputjps;
  THTensor_(resize2d)(alistj_ptr, partCounted, (long)(partCounted/2) * partCounted);
  real * alistj  = THTensor_(data)(alistj_ptr);
  long   alistj_s0 = THTensor_(stride)(alistj_ptr, 0);
  long   alistj_s1 = THTensor_(stride)(alistj_ptr, 1);
  real*  alistjp;
  
  THTensor_(resize4d)(Angoutputk_ptr, partCounted, (long)(partCounted/2) * partCounted, num_ang, 3);
  real * Angoutputk  = THTensor_(data)(Angoutputk_ptr);
  long   Angoutputk_s0 = THTensor_(stride)(Angoutputk_ptr, 0);
  long   Angoutputk_s1 = THTensor_(stride)(Angoutputk_ptr, 1);
  long   Angoutputk_s2 = THTensor_(stride)(Angoutputk_ptr, 2);
  long   Angoutputk_s3 = THTensor_(stride)(Angoutputk_ptr, 3);
  real*  Angoutputkp;
  real*  Angoutputkpj;
  real*  Angoutputkps;
  THTensor_(resize2d)(alistk_ptr, partCounted, (long)(partCounted/2) * partCounted);
  real * alistk  = THTensor_(data)(alistk_ptr);
  long   alistk_s0 = THTensor_(stride)(alistk_ptr, 0);
  long   alistk_s1 = THTensor_(stride)(alistk_ptr, 1);
  real*  alistkp;

  real* input_data  = THTensor_(data)(input);
  long  input_s0 = THTensor_(stride)(input, 0);
  long  input_s1 = THTensor_(stride)(input, 1);
  real* inputp;

  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long   PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);

  real* rad_eta  = THTensor_(data)(rad_eta_ptr);
  long  rad_eta_s = THTensor_(stride)(rad_eta_ptr, 0);
  real* ang_eta  = THTensor_(data)(ang_eta_ptr);
  long  ang_eta_s = THTensor_(stride)(ang_eta_ptr, 0);
  real* lambda  = THTensor_(data)(lambda_ptr);
  long  lambda_s = THTensor_(stride)(lambda_ptr, 0);
  real* zeta  = THTensor_(data)(zeta_ptr);
  long  zeta_s = THTensor_(stride)(zeta_ptr, 0);

  real* numOfPartNeighList  = THTensor_(data)(numOfPartNeighList_ptr);
  long  numOfPartNeighList_s = THTensor_(stride)(numOfPartNeighList_ptr, 0);
  real* neighListOfParts  = THTensor_(data)(neighListOfParts_ptr);
  long  neighListOfParts_s0 = THTensor_(stride)(neighListOfParts_ptr, 0);
  long  neighListOfParts_s1 = THTensor_(stride)(neighListOfParts_ptr, 1);
  real* partNLp;
  real* Rij_list  = THTensor_(data)(Rij_list_ptr);
  long  Rij_list_s0 = THTensor_(stride)(Rij_list_ptr, 0);
  long  Rij_list_s1 = THTensor_(stride)(Rij_list_ptr, 1);
  long  Rij_list_s2 = THTensor_(stride)(Rij_list_ptr, 2);
  real* Rijpi;
  real* Rijpj;
  real* Rijpk;
  
  long i,j,k,jj,kk,d;
  long numOfPartNeigh;
  real Rsqij,Rsqik,Rsqjk;
  real cutsq = cutoff * cutoff; 
  if(neighbin > 0.0) cutsq = cutsq + (neighbin * neighbin);
  real Rij[3],Rik[3],Rjk[3];
  real fcutall, fcutik, fcutij, fcutjk;
  real fcut_gradi_ij, fcut_gradi_ik, fcut_gradj_jk;
  real thetagradi[3],thetagradj[3],thetagradk[3];
  real costheta, costhetaVALUE;
  real power1, power2, power3, vexp1, vexp2, vexp3, vexp4;
  real powksi1, expon, angexp;
  real RijRik, dijdijV, dijdij, dijdikV, dikdikV;
  real ffi[3],ffj[3],ffk[3];
  long offset, symi;
  real pi=3.1415926535897932384626433832795028841971694;
  real dij,djk,dik,dall;
  long ityp;
  long jtyp;
  long ktyp;
  long rnum;
  long anum;

 for(i=0;i<partCounted;i++){
    numOfPartNeigh = (long)numOfPartNeighList[i * numOfPartNeighList_s];
    partNLp = neighListOfParts + i * neighListOfParts_s0;
    ityp = (long)particleSpecies[i * particleSpecies_s];
    outputpi = output_data + i * output_s0;
    Radoutputip = Radoutputi + i * Radoutputi_s0;
    Angoutputip = Angoutputi + i * Angoutputi_s0;
    Angoutputjp = Angoutputj + i * Angoutputj_s0;
    Angoutputkp = Angoutputk + i * Angoutputk_s0;
    rlistjp = rlistj + i * rlistj_s0;
    alistjp = alistj + i * alistj_s0;
    alistkp = alistk + i * alistk_s0;
    inputp = input_data + (long)PartListOffset[i * PartListOffset_s] * input_s0;
    Rijpi = Rij_list + i * Rij_list_s0;

    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = (long)partNLp[jj * neighListOfParts_s1];
      Rsqij = 0.0;
      Rijpj = Rijpi + jj * Rij_list_s1;
      for (d = 0; d < 3; ++d) {
        Rij[d] =  Rijpj[d * Rij_list_s2]; 
        Rsqij +=  Rij[d] * Rij[d]; 
      } /* End of Rij[k] loop */

      if (Rsqij < cutsq && Rsqij > 0.000000001) {
        for (d = 0; d < 3; d++)  ffi[d] = 0.0;
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        rnum = (long)radlistnum[i * radlistnum_s];
        rlistjp[rnum * rlistj_s1] = j;
        offset = (jtyp-1) * num_rad;
        outputpj = output_data + j * output_s0;
        Radoutputipj = Radoutputip + rnum * Radoutputi_s1;

#if defined(TH_REAL_IS_FLOAT)
        dij = sqrtf(Rsqij);
        dijdij = dij * dij;
        fcutij = (dij < cutoff) ? 0.5 * (cosf(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        fcut_gradi_ij = (dij < cutoff) ? -(0.5 * pi/cutoff ) * sinf(pi * dij/cutoff ) / dij : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            Radoutputips = Radoutputipj + symi * Radoutputi_s2;
            vexp2 = inputp[(offset + symi) * input_s1] * expf( -rad_eta[rad_eta_s * symi] * dijdij); 
            vexp1 = -2.0 * rad_eta[rad_eta_s * symi] * fcutij * vexp2; 
            vexp4 = expf( -rad_eta[rad_eta_s * symi] * dijdij); 
            vexp3 = -2.0 * rad_eta[rad_eta_s * symi] * fcutij * vexp4; 
            for (d = 0; d < 3; d++) { ffi[d] += ( vexp1 + vexp2 * fcut_gradi_ij ) * Rij[d]; 
                                      Radoutputips[d * Radoutputi_s3] = ( vexp3 + vexp4 * fcut_gradi_ij ) * Rij[d]; } 
        }
#else
        dij = sqrt(Rsqij);
        dijdij = dij * dij;
        fcutij = (dij < cutoff) ? 0.5 * (cos(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        fcut_gradi_ij = (dij < cutoff) ? -(0.5 * pi/cutoff ) * sin(pi * dij/cutoff ) / dij : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            Radoutputips = Radoutputipj + symi * Radoutputi_s2;
            vexp2 = inputp[(offset + symi) * input_s1] * exp( -rad_eta[rad_eta_s * symi] * dijdij); 
            vexp1 = -2.0 * rad_eta[rad_eta_s * symi] * fcutij * vexp2; 
            vexp4 =  exp( -rad_eta[rad_eta_s * symi] * dijdij); 
            vexp3 = -2.0 * rad_eta[rad_eta_s * symi] * fcutij * vexp4; 
            for (d = 0; d < 3; d++) { ffi[d] += ( vexp1 + vexp2 * fcut_gradi_ij ) * Rij[d]; 
                                      Radoutputips[d * Radoutputi_s3] = ( vexp3 + vexp4 * fcut_gradi_ij ) * Rij[d]; }
        }
#endif
        for (d = 0; d < 3; d++) {  outputpi[d * output_s1] -= ffi[d]; 
                                   outputpj[d * output_s1] += ffi[d]; }

        for (kk = 0; kk < numOfPartNeigh; ++kk) {
           if (jj==kk) continue;
           k = (long)partNLp[kk * neighListOfParts_s1];
           if (j==k && i==k) continue;

          Rsqik = 0.0;
          Rsqjk = 0.0;
          Rijpk = Rijpi + kk * Rij_list_s1;
          for (d = 0; d < 3; ++d) {
            Rik[d] = Rijpk[d * Rij_list_s2]; 
            Rjk[d] = Rik[d] - Rij[d]; 
            Rsqik += Rik[d]*Rik[d]; 
            Rsqjk += Rjk[d]*Rjk[d]; 
          } /* End of Rij[k] loop */
        
          if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.000000001 && Rsqjk > 0.000000001) {
          //if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.00001 && Rsqjk > 0.00001) {
             outputpk = output_data + k * output_s0;
             anum = anglistnum[i * anglistnum_s];
             Angoutputipj = Angoutputip + anum * Angoutputi_s1;
             Angoutputjpj = Angoutputjp + anum * Angoutputj_s1;
             Angoutputkpj = Angoutputkp + anum * Angoutputk_s1;
             alistjp[anum * alistj_s1] = j;
             alistkp[anum * alistk_s1] = k;
             ktyp =  (long)particleSpecies[k * particleSpecies_s];
             offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);
             RijRik = 0.0;
             for (d = 0; d < 3; d++) {   ffi[d] = 0.0; ffj[d] = 0.0; ffk[d] = 0.0; RijRik += Rij[d]*Rik[d]; }

#if defined(TH_REAL_IS_FLOAT)
             dik = sqrtf(Rsqik);
             djk = sqrtf(Rsqjk);
             fcutjk = (djk < cutoff) ? 0.5 * (cosf(pi * djk / cutoff ) + 1.0) : 0.0 ; 
             fcutik = (dik < cutoff) ? 0.5 * (cosf(pi * dik / cutoff ) + 1.0) : 0.0 ; 
             fcut_gradi_ik = (dik < cutoff) ? -(0.5 * pi/cutoff ) * sinf(pi * dik/cutoff ) / dik : 0.0 ; 
             fcut_gradj_jk = (djk < cutoff) ? -(0.5 * pi/cutoff ) * sinf(pi * djk/cutoff ) / djk : 0.0 ; 
#else
             dik = sqrt(Rsqik);
             djk = sqrt(Rsqjk);
             fcutjk = (djk < cutoff) ? 0.5 * (cos(pi * djk / cutoff ) + 1.0) : 0.0 ; 
             fcutik = (dik < cutoff) ? 0.5 * (cos(pi * dik / cutoff ) + 1.0) : 0.0 ; 
             fcut_gradi_ik = (dik < cutoff) ? -(0.5 * pi/cutoff ) * sin(pi * dik/cutoff ) / dik : 0.0 ; 
             fcut_gradj_jk = (djk < cutoff) ? -(0.5 * pi/cutoff ) * sin(pi * djk/cutoff ) / djk : 0.0 ; 
#endif

             dijdijV = 1.0/(dij * dij);
             dikdikV = 1.0/(dik * dik);
             dijdikV = 1.0/(dij * dik);
             for(d=0;d<3;d++){
                thetagradi[d] = ( ( Rij[d] + Rik[d] ) - RijRik * ( Rij[d] * dijdijV + Rik[d] * dikdikV ) ) * dijdikV;
                thetagradj[d] = ( RijRik * Rij[d] * dijdijV - Rik[d] ) * dijdikV;
                thetagradk[d] = ( RijRik * Rik[d] * dikdikV - Rij[d] ) * dijdikV;
             } 

             fcutall = fcutik * fcutij * fcutjk;
             dall = dik*dik + dijdij + djk*djk ;
             costhetaVALUE = RijRik * dijdikV;

             for (symi=0;symi<num_ang;symi++) { 
                 Angoutputips = Angoutputipj + symi * Angoutputi_s2;
                 Angoutputjps = Angoutputjpj + symi * Angoutputj_s2;
                 Angoutputkps = Angoutputkpj + symi * Angoutputk_s2;
                 costheta = 1.0 + lambda[lambda_s * symi] * costhetaVALUE;
#if defined(TH_REAL_IS_FLOAT)
                 expon   = expf( -ang_eta[ang_eta_s * symi] * dall);
                 power1  = powf(costheta, zeta[zeta_s * symi]);
                 power2  = inputp[(symi+offset) * input_s1] * powf(2.0,1.0 - zeta[zeta_s * symi]);
                 power3  = powf(2.0,1.0 - zeta[zeta_s * symi]);
                 powksi1 = powf(costheta,(zeta[zeta_s * symi]-1.0)) * zeta[zeta_s * symi] * lambda[lambda_s * symi] * expon * fcutall;
#else
                 expon   = exp( -ang_eta[ang_eta_s * symi] * dall);
                 power1  = pow(costheta, zeta[zeta_s * symi]);
                 power2  = inputp[(symi+offset) * input_s1] * pow(2.0, 1.0 - zeta[zeta_s * symi]);
                 power3  = pow(2.0, 1.0 - zeta[zeta_s * symi]);
                 powksi1 = pow(costheta,(zeta[zeta_s * symi]-1.0))  * zeta[zeta_s * symi] * lambda[lambda_s * symi] * expon * fcutall;
#endif
                 angexp = -ang_eta[ang_eta_s * symi] * 2.0 * expon * fcutall;
                 for(d=0;d<3;d++){
                    ffi[d] -=  power2 * ( ( thetagradi[d] * powksi1 ) +  
                               power1 * ( ( (  Rij[d] + Rik[d] ) * angexp ) + 
                                          ( expon * fcutjk * (  fcut_gradi_ij * Rij[d] * fcutik + fcutij *  fcut_gradi_ik * Rik[d] ) )
                                        )
                               );
                    Angoutputips[d * Angoutputi_s3] = - power3 * 
                               ( ( thetagradi[d] * powksi1 ) +  
                               power1 * ( ( (  Rij[d] + Rik[d] ) * angexp ) + 
                                          ( expon * fcutjk * (  fcut_gradi_ij * Rij[d] * fcutik + fcutij *  fcut_gradi_ik * Rik[d] ) )
                                        )
                               );
                    ffj[d] -=  power2 * ( ( thetagradj[d] * powksi1 ) +  
                               power1 * ( ( ( -Rij[d] + Rjk[d] ) * angexp ) + 
                                          ( expon * fcutik * ( -fcut_gradi_ij * Rij[d] * fcutjk + fcutij *  fcut_gradj_jk * Rjk[d] ) )
                                        )
                               );
                    Angoutputjps[d * Angoutputj_s3] = - power3 * 
                               ( ( thetagradj[d] * powksi1 ) +  
                               power1 * ( ( ( -Rij[d] + Rjk[d] ) * angexp ) + 
                                          ( expon * fcutik * ( -fcut_gradi_ij * Rij[d] * fcutjk + fcutij *  fcut_gradj_jk * Rjk[d] ) )
                                        )
                               );
                    ffk[d] -=  power2 * ( ( thetagradk[d] * powksi1 ) +  
                               power1 * ( ( ( -Rik[d] - Rjk[d] ) * angexp ) + 
                                          ( expon * fcutij * ( -fcut_gradi_ik * Rik[d] * fcutjk - fcutik *  fcut_gradj_jk * Rjk[d] ) )
                                        )
                               );
                    Angoutputkps[d * Angoutputk_s3] = - power3 * 
                               ( ( thetagradk[d] * powksi1 ) +  
                               power1 * ( ( ( -Rik[d] - Rjk[d] ) * angexp ) + 
                                          ( expon * fcutij * ( -fcut_gradi_ik * Rik[d] * fcutjk - fcutik *  fcut_gradj_jk * Rjk[d] ) )
                                        )
                               );
                 }
             }

             for (d = 0; d < 3; d++) {   outputpi[output_s1 * d] += ffi[d];
                                         outputpj[output_s1 * d] += ffj[d]; 
                                         outputpk[output_s1 * d] += ffk[d]; }
             anglistnum[i * anglistnum_s]++;
          } /* End of if inside cutoff, Rsqik < cutoff */
        }  /* loop on kk */
        radlistnum[i * radlistnum_s]++;
      } /* End of if inside cutoff, Rsqij < cutoff */
    }  /* loop on jj */
  }  /* infinite while loop on i (terminated by break statements above) */

  lua_getfield(L, 1, "output");
  return 1;
}


static int poten_(Energy_calculateAllnnForces)(lua_State *L)
{
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  long threaded = luaL_optnumber(L, 3, 0);
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_syms = luaT_getfieldchecknumber(L, 1, "numSyms");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);
  THTensor * Radoutputi_ptr = luaT_getfieldcheckudata(L, 1, "RadOutputi", torch_Tensor);
  THTensor * Angoutputi_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputi", torch_Tensor);
  THTensor * Angoutputj_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputj", torch_Tensor);
  THTensor * Angoutputk_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputk", torch_Tensor);
  THTensor * rlistj_ptr = luaT_getfieldcheckudata(L, 1, "RadListj", torch_Tensor);
  THTensor * alistj_ptr = luaT_getfieldcheckudata(L, 1, "AngListj", torch_Tensor);
  THTensor * alistk_ptr = luaT_getfieldcheckudata(L, 1, "AngListk", torch_Tensor);
  THTensor * radlistnum_ptr = luaT_getfieldcheckudata(L, 1, "RadListNum", torch_Tensor);
  THTensor * anglistnum_ptr = luaT_getfieldcheckudata(L, 1, "AngListNum", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  real* radlistnum = THTensor_(data)(radlistnum_ptr);
  long  radlistnum_s = THTensor_(stride)(radlistnum_ptr, 0);
  real* anglistnum = THTensor_(data)(anglistnum_ptr);
  long  anglistnum_s = THTensor_(stride)(anglistnum_ptr, 0);
  
  THTensor_(resize2d)(output, partCounted, 3);
  THTensor_(zero)(output);
  real* output_data = THTensor_(data)(output);
  long  output_s0 = THTensor_(stride)(output, 0);
  long  output_s1 = THTensor_(stride)(output, 1);
  
  real * Radoutputi  = THTensor_(data)(Radoutputi_ptr);
  long   Radoutputi_s0 = THTensor_(stride)(Radoutputi_ptr, 0);
  long   Radoutputi_s1 = THTensor_(stride)(Radoutputi_ptr, 1);
  long   Radoutputi_s2 = THTensor_(stride)(Radoutputi_ptr, 2);
  long   Radoutputi_s3 = THTensor_(stride)(Radoutputi_ptr, 3);
  real * rlistj  = THTensor_(data)(rlistj_ptr);
  long   rlistj_s0 = THTensor_(stride)(rlistj_ptr, 0);
  long   rlistj_s1 = THTensor_(stride)(rlistj_ptr, 1);
  
  real * Angoutputi  = THTensor_(data)(Angoutputi_ptr);
  long   Angoutputi_s0 = THTensor_(stride)(Angoutputi_ptr, 0);
  long   Angoutputi_s1 = THTensor_(stride)(Angoutputi_ptr, 1);
  long   Angoutputi_s2 = THTensor_(stride)(Angoutputi_ptr, 2);
  long   Angoutputi_s3 = THTensor_(stride)(Angoutputi_ptr, 3);
  
  real * Angoutputj  = THTensor_(data)(Angoutputj_ptr);
  long   Angoutputj_s0 = THTensor_(stride)(Angoutputj_ptr, 0);
  long   Angoutputj_s1 = THTensor_(stride)(Angoutputj_ptr, 1);
  long   Angoutputj_s2 = THTensor_(stride)(Angoutputj_ptr, 2);
  long   Angoutputj_s3 = THTensor_(stride)(Angoutputj_ptr, 3);
  real * alistj  = THTensor_(data)(alistj_ptr);
  long   alistj_s0 = THTensor_(stride)(alistj_ptr, 0);
  long   alistj_s1 = THTensor_(stride)(alistj_ptr, 1);
  
  real * Angoutputk  = THTensor_(data)(Angoutputk_ptr);
  long   Angoutputk_s0 = THTensor_(stride)(Angoutputk_ptr, 0);
  long   Angoutputk_s1 = THTensor_(stride)(Angoutputk_ptr, 1);
  long   Angoutputk_s2 = THTensor_(stride)(Angoutputk_ptr, 2);
  long   Angoutputk_s3 = THTensor_(stride)(Angoutputk_ptr, 3);
  real * alistk  = THTensor_(data)(alistk_ptr);
  long   alistk_s0 = THTensor_(stride)(alistk_ptr, 0);
  long   alistk_s1 = THTensor_(stride)(alistk_ptr, 1);

  real* input_data  = THTensor_(data)(input);
  long  input_s0 = THTensor_(stride)(input, 0);
  long  input_s1 = THTensor_(stride)(input, 1);

  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long   PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);

  long i;
  
if(threaded>0){
 
  THTensor *output_private = THTensor_(new)();

#pragma omp parallel private(i) if(partCounted > 64)
{
 const int ThreadId = omp_get_thread_num();
 const int Threads = omp_get_num_threads();
 long offset, symi;
 long jj, kk, j, k;
 long ityp, jtyp, ktyp;
 int d;
 real *inputp;
 real *inp;
 real *rlistjp;
 real *alistjp;
 real *alistkp;
 real *Radoutputip;
 real *Radoutputipj;
 real *Radoutputips;
 real *Angoutputip;
 real *Angoutputipj;
 real *Angoutputips;
 real *Angoutputjp;
 real *Angoutputjpj;
 real *Angoutputjps;
 real *Angoutputkp;
 real *Angoutputkpj;
 real *Angoutputkps;

 #pragma omp single
 {
   THTensor_(resize2d)(output_private, Threads*partCounted, 3);
   THTensor_(zero)(output_private);
 }
 
 real *op = THTensor_(data)(output_private);
 long  op_s0 = THTensor_(stride)(output_private, 0);
 long  op_s1 = THTensor_(stride)(output_private, 1);
 real *opi;
 real *opj;
 real *opk;

 #pragma omp for schedule(static) 
 for(i=0;i<partCounted;i++){
    ityp = (long)particleSpecies[i * particleSpecies_s];
    opi = op + (partCounted * ThreadId + i) * op_s0;
    Radoutputip = Radoutputi + i * Radoutputi_s0;
    Angoutputip = Angoutputi + i * Angoutputi_s0;
    Angoutputjp = Angoutputj + i * Angoutputj_s0;
    Angoutputkp = Angoutputk + i * Angoutputk_s0;
    rlistjp = rlistj + i * rlistj_s0;
    alistjp = alistj + i * alistj_s0;
    alistkp = alistk + i * alistk_s0;
    inputp = input_data + (long)PartListOffset[i * PartListOffset_s] * input_s0;

    for (jj = 0; jj < radlistnum[i * radlistnum_s]; ++jj) {
        j = rlistjp[jj * rlistj_s1];
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        offset = (jtyp-1) * num_rad;
        opj = op + (partCounted * ThreadId + j) * op_s0;
        Radoutputipj = Radoutputip + jj * Radoutputi_s1;
        inp = inputp + offset * input_s1;

        for (symi=0;symi<num_rad;symi++) { 
            Radoutputips = Radoutputipj + symi * Radoutputi_s2;
            for (d = 0; d < 3; d++) {  opi[d * op_s1] -= inp[symi * input_s1] * Radoutputips[d * Radoutputi_s3];
                                       opj[d * op_s1] += inp[symi * input_s1] * Radoutputips[d * Radoutputi_s3]; }
        } /* loop on symi */

    }  /* loop on jj */

    for (kk = 0; kk < anglistnum[i * anglistnum_s]; ++kk) {
        j = alistjp[kk * alistj_s1];
        k = alistkp[kk * alistk_s1];
        opj = op + (partCounted * ThreadId + j) * op_s0;
        opk = op + (partCounted * ThreadId + k) * op_s0;
        Angoutputipj = Angoutputip + kk * Angoutputi_s1;
        Angoutputjpj = Angoutputjp + kk * Angoutputj_s1;
        Angoutputkpj = Angoutputkp + kk * Angoutputk_s1;
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        ktyp =  (long)particleSpecies[k * particleSpecies_s];
        offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);
        inp = inputp + offset * input_s1;

        for (symi=0;symi<num_ang;symi++) { 
            Angoutputips = Angoutputipj + symi * Angoutputi_s2;
            Angoutputjps = Angoutputjpj + symi * Angoutputj_s2;
            Angoutputkps = Angoutputkpj + symi * Angoutputk_s2;
            for (d = 0; d < 3; d++) {   opi[d * op_s1] += inp[symi * input_s1] * Angoutputips[d * Angoutputi_s3]; 
                                        opj[d * op_s1] += inp[symi * input_s1] * Angoutputjps[d * Angoutputj_s3]; 
                                        opk[d * op_s1] += inp[symi * input_s1] * Angoutputkps[d * Angoutputk_s3]; }
        } /* loop on symi */

    }  /* loop on kk */

  }  /* infinite while loop on i (terminated by break statements above) */
 
   //THTensor_(resize3d)(output_private, Threads, partCounted, 3);
 #pragma omp single
 {
   real *opt;
   for(i=0;i<Threads;i++){
      for (jj = 0; jj < partCounted; ++jj) {
          opi = op + (partCounted * i + jj) * op_s0;
          opt = output_data + jj * output_s0;
          for (d = 0; d < 3; d++) opt[d * output_s1] += opi[d * op_s1];
      }
   }
 }

} // Parallel off
  THTensor_(free)(output_private); 

} else {
 
 long offset, symi;
 long jj, kk, j, k;
 long ityp, jtyp, ktyp;
 int d;
 real *inputp;
 real *inp;
 real *rlistjp;
 real *alistjp;
 real *alistkp;
 real *Radoutputip;
 real *Radoutputipj;
 real *Radoutputips;
 real *Angoutputip;
 real *Angoutputipj;
 real *Angoutputips;
 real *Angoutputjp;
 real *Angoutputjpj;
 real *Angoutputjps;
 real *Angoutputkp;
 real *Angoutputkpj;
 real *Angoutputkps;
 real *opi;
 real *opj;
 real *opk;

 for(i=0;i<partCounted;i++){
    ityp = (long)particleSpecies[i * particleSpecies_s];
    opi = output_data + i * output_s0;
    Radoutputip = Radoutputi + i * Radoutputi_s0;
    Angoutputip = Angoutputi + i * Angoutputi_s0;
    Angoutputjp = Angoutputj + i * Angoutputj_s0;
    Angoutputkp = Angoutputk + i * Angoutputk_s0;
    rlistjp = rlistj + i * rlistj_s0;
    alistjp = alistj + i * alistj_s0;
    alistkp = alistk + i * alistk_s0;
    inputp = input_data + (long)PartListOffset[i * PartListOffset_s] * input_s0;

    for (jj = 0; jj < radlistnum[i * radlistnum_s]; ++jj) {
        j = rlistjp[jj * rlistj_s1];
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        offset = (jtyp-1) * num_rad;
        opj = output_data + j * output_s0;
        Radoutputipj = Radoutputip + jj * Radoutputi_s1;
        inp = inputp + offset * input_s1;

        for (symi=0;symi<num_rad;symi++) { 
            Radoutputips = Radoutputipj + symi * Radoutputi_s2;
            for (d = 0; d < 3; d++) {  opi[d * output_s1] -= inp[symi * input_s1] * Radoutputips[d * Radoutputi_s3];
                                       opj[d * output_s1] += inp[symi * input_s1] * Radoutputips[d * Radoutputi_s3]; }
        } /* loop on symi */

    }  /* loop on jj */

    for (kk = 0; kk < anglistnum[i * anglistnum_s]; ++kk) {
        j = alistjp[kk * alistj_s1];
        k = alistkp[kk * alistk_s1];
        opj = output_data + j * output_s0;
        opk = output_data + k * output_s0;
        Angoutputipj = Angoutputip + kk * Angoutputi_s1;
        Angoutputjpj = Angoutputjp + kk * Angoutputj_s1;
        Angoutputkpj = Angoutputkp + kk * Angoutputk_s1;
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        ktyp =  (long)particleSpecies[k * particleSpecies_s];
        offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);
        inp = inputp + offset * input_s1;

        for (symi=0;symi<num_ang;symi++) { 
            Angoutputips = Angoutputipj + symi * Angoutputi_s2;
            Angoutputjps = Angoutputjpj + symi * Angoutputj_s2;
            Angoutputkps = Angoutputkpj + symi * Angoutputk_s2;
            for (d = 0; d < 3; d++) {   opi[d * output_s1] += inp[symi * input_s1] * Angoutputips[d * Angoutputi_s3]; 
                                        opj[d * output_s1] += inp[symi * input_s1] * Angoutputjps[d * Angoutputj_s3]; 
                                        opk[d * output_s1] += inp[symi * input_s1] * Angoutputkps[d * Angoutputk_s3]; }
        } /* loop on symi */

    }  /* loop on kk */

  }  /* infinite while loop on i (terminated by break statements above) */
 
} /* if threaded which is an option */

  lua_getfield(L, 1, "output");
  return 1;
}


static int poten_(Energy_calculateAllForces)(lua_State *L)
{
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  long threaded = luaL_optnumber(L, 3, 0);
  long partCounted = luaT_getfieldchecknumber(L, 1, "numParts");
  long num_rad = luaT_getfieldchecknumber(L, 1, "numRadial");
  long num_ang = luaT_getfieldchecknumber(L, 1, "numAngular");
  long num_syms = luaT_getfieldchecknumber(L, 1, "numSyms");
  long num_typ = luaT_getfieldchecknumber(L, 1, "numTypes");
  THTensor * particleSpecies_ptr = luaT_getfieldcheckudata(L, 1, "partSpecies", torch_Tensor);
  THTensor * PartListOffset_ptr = luaT_getfieldcheckudata(L, 1, "outputList", torch_Tensor);
  THTensor * Radoutputi_ptr = luaT_getfieldcheckudata(L, 1, "RadOutputi", torch_Tensor);
  THTensor * Angoutputi_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputi", torch_Tensor);
  THTensor * Angoutputj_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputj", torch_Tensor);
  THTensor * Angoutputk_ptr = luaT_getfieldcheckudata(L, 1, "AngOutputk", torch_Tensor);
  THTensor * rlistj_ptr = luaT_getfieldcheckudata(L, 1, "RadListj", torch_Tensor);
  THTensor * alistj_ptr = luaT_getfieldcheckudata(L, 1, "AngListj", torch_Tensor);
  THTensor * alistk_ptr = luaT_getfieldcheckudata(L, 1, "AngListk", torch_Tensor);
  THTensor * radlistnum_ptr = luaT_getfieldcheckudata(L, 1, "RadListNum", torch_Tensor);
  THTensor * anglistnum_ptr = luaT_getfieldcheckudata(L, 1, "AngListNum", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  real* radlistnum = THTensor_(data)(radlistnum_ptr);
  long  radlistnum_s = THTensor_(stride)(radlistnum_ptr, 0);
  real* anglistnum = THTensor_(data)(anglistnum_ptr);
  long  anglistnum_s = THTensor_(stride)(anglistnum_ptr, 0);
  
  THTensor_(resize2d)(output, partCounted, 3);
  THTensor_(zero)(output);
  real* output_data = THTensor_(data)(output);
  long  output_s0 = THTensor_(stride)(output, 0);
  long  output_s1 = THTensor_(stride)(output, 1);
  
  real * Radoutputi  = THTensor_(data)(Radoutputi_ptr);
  long   Radoutputi_s0 = THTensor_(stride)(Radoutputi_ptr, 0);
  long   Radoutputi_s1 = THTensor_(stride)(Radoutputi_ptr, 1);
  long   Radoutputi_s2 = THTensor_(stride)(Radoutputi_ptr, 2);
  long   Radoutputi_s3 = THTensor_(stride)(Radoutputi_ptr, 3);
  real * rlistj  = THTensor_(data)(rlistj_ptr);
  long   rlistj_s0 = THTensor_(stride)(rlistj_ptr, 0);
  long   rlistj_s1 = THTensor_(stride)(rlistj_ptr, 1);
  
  real * Angoutputi  = THTensor_(data)(Angoutputi_ptr);
  long   Angoutputi_s0 = THTensor_(stride)(Angoutputi_ptr, 0);
  long   Angoutputi_s1 = THTensor_(stride)(Angoutputi_ptr, 1);
  long   Angoutputi_s2 = THTensor_(stride)(Angoutputi_ptr, 2);
  long   Angoutputi_s3 = THTensor_(stride)(Angoutputi_ptr, 3);
  
  real * Angoutputj  = THTensor_(data)(Angoutputj_ptr);
  long   Angoutputj_s0 = THTensor_(stride)(Angoutputj_ptr, 0);
  long   Angoutputj_s1 = THTensor_(stride)(Angoutputj_ptr, 1);
  long   Angoutputj_s2 = THTensor_(stride)(Angoutputj_ptr, 2);
  long   Angoutputj_s3 = THTensor_(stride)(Angoutputj_ptr, 3);
  real * alistj  = THTensor_(data)(alistj_ptr);
  long   alistj_s0 = THTensor_(stride)(alistj_ptr, 0);
  long   alistj_s1 = THTensor_(stride)(alistj_ptr, 1);
  
  real * Angoutputk  = THTensor_(data)(Angoutputk_ptr);
  long   Angoutputk_s0 = THTensor_(stride)(Angoutputk_ptr, 0);
  long   Angoutputk_s1 = THTensor_(stride)(Angoutputk_ptr, 1);
  long   Angoutputk_s2 = THTensor_(stride)(Angoutputk_ptr, 2);
  long   Angoutputk_s3 = THTensor_(stride)(Angoutputk_ptr, 3);
  real * alistk  = THTensor_(data)(alistk_ptr);
  long   alistk_s0 = THTensor_(stride)(alistk_ptr, 0);
  long   alistk_s1 = THTensor_(stride)(alistk_ptr, 1);

  real* input_data  = THTensor_(data)(input);
  long  input_s0 = THTensor_(stride)(input, 0);
  long  input_s1 = THTensor_(stride)(input, 1);

  real * particleSpecies = THTensor_(data)(particleSpecies_ptr);
  long   particleSpecies_s = THTensor_(stride)(particleSpecies_ptr, 0);
  real * PartListOffset   = THTensor_(data)(PartListOffset_ptr);
  long   PartListOffset_s = THTensor_(stride)(PartListOffset_ptr, 0);

  long i;
  
if(threaded>0){
 
  THTensor *output_private = THTensor_(new)();

#pragma omp parallel private(i) if(partCounted > 64)
{
 const int ThreadId = omp_get_thread_num();
 const int Threads = omp_get_num_threads();
 long offset, symi;
 long jj, kk, j, k;
 long ityp, jtyp, ktyp;
 int d;
 real val;
 real *inputp;
 real *rlistjp;
 real *alistjp;
 real *alistkp;
 real *Radoutputip;
 real *Radoutputipj;
 real *Radoutputips;
 real *Angoutputip;
 real *Angoutputipj;
 real *Angoutputips;
 real *Angoutputjp;
 real *Angoutputjpj;
 real *Angoutputjps;
 real *Angoutputkp;
 real *Angoutputkpj;
 real *Angoutputkps;

 #pragma omp single
 {
   THTensor_(resize2d)(output_private, Threads*partCounted, 3);
   THTensor_(zero)(output_private);
 }
 
 real *op = THTensor_(data)(output_private);
 long  op_s0 = THTensor_(stride)(output_private, 0);
 long  op_s1 = THTensor_(stride)(output_private, 1);
 real *opi;
 real *opj;
 real *opk;

 #pragma omp for schedule(static) 
 for(i=0;i<partCounted;i++){
    ityp = (long)particleSpecies[i * particleSpecies_s];
    opi = op + (partCounted * ThreadId + i) * op_s0;
    Radoutputip = Radoutputi + i * Radoutputi_s0;
    Angoutputip = Angoutputi + i * Angoutputi_s0;
    Angoutputjp = Angoutputj + i * Angoutputj_s0;
    Angoutputkp = Angoutputk + i * Angoutputk_s0;
    rlistjp = rlistj + i * rlistj_s0;
    alistjp = alistj + i * alistj_s0;
    alistkp = alistk + i * alistk_s0;
    inputp = input_data + (long)PartListOffset[i * PartListOffset_s] * input_s0;

    for (jj = 0; jj < radlistnum[i * radlistnum_s]; ++jj) {
        j = rlistjp[jj * rlistj_s1];
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        offset = (jtyp-1) * num_rad;
        opj = op + (partCounted * ThreadId + j) * op_s0;
        Radoutputipj = Radoutputip + jj * Radoutputi_s1;

        for (symi=0;symi<num_rad;symi++) { 
            Radoutputips = Radoutputipj + symi * Radoutputi_s2;
            val = inputp[(offset + symi) * input_s1]; 
            for (d = 0; d < 3; d++) {  opi[d * op_s1] -= val * Radoutputips[d * Radoutputi_s3];
                                       opj[d * op_s1] += val * Radoutputips[d * Radoutputi_s3]; }
        } /* loop on symi */

    }  /* loop on jj */

    for (kk = 0; kk < anglistnum[i * anglistnum_s]; ++kk) {
        j = alistjp[kk * alistj_s1];
        k = alistkp[kk * alistk_s1];
        opj = op + (partCounted * ThreadId + j) * op_s0;
        opk = op + (partCounted * ThreadId + k) * op_s0;
        Angoutputipj = Angoutputip + kk * Angoutputi_s1;
        Angoutputjpj = Angoutputjp + kk * Angoutputj_s1;
        Angoutputkpj = Angoutputkp + kk * Angoutputk_s1;
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        ktyp =  (long)particleSpecies[k * particleSpecies_s];
        offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);

        for (symi=0;symi<num_ang;symi++) { 
            Angoutputips = Angoutputipj + symi * Angoutputi_s2;
            Angoutputjps = Angoutputjpj + symi * Angoutputj_s2;
            Angoutputkps = Angoutputkpj + symi * Angoutputk_s2;
            val = inputp[(offset + symi) * input_s1]; 
            for (d = 0; d < 3; d++) {   opi[d * op_s1] += val * Angoutputips[d * Angoutputi_s3]; 
                                        opj[d * op_s1] += val * Angoutputjps[d * Angoutputj_s3]; 
                                        opk[d * op_s1] += val * Angoutputkps[d * Angoutputk_s3]; }
        } /* loop on symi */

    }  /* loop on kk */

  }  /* infinite while loop on i (terminated by break statements above) */
 
   //THTensor_(resize3d)(output_private, Threads, partCounted, 3);
 #pragma omp single
 {
   real *opt;
   for(i=0;i<Threads;i++){
      for (jj = 0; jj < partCounted; ++jj) {
          opi = op + (partCounted * i + jj) * op_s0;
          opt = output_data + jj * output_s0;
          for (d = 0; d < 3; d++) opt[d * output_s1] += opi[d * op_s1];
      }
   }
 }

} // Parallel off
  THTensor_(free)(output_private); 

} else {
 
 long offset, symi;
 long jj, kk, j, k;
 long ityp, jtyp, ktyp;
 int d;
 real val;
 real *inputp;
 real *rlistjp;
 real *alistjp;
 real *alistkp;
 real *Radoutputip;
 real *Radoutputipj;
 real *Radoutputips;
 real *Angoutputip;
 real *Angoutputipj;
 real *Angoutputips;
 real *Angoutputjp;
 real *Angoutputjpj;
 real *Angoutputjps;
 real *Angoutputkp;
 real *Angoutputkpj;
 real *Angoutputkps;
 real *opi;
 real *opj;
 real *opk;

 for(i=0;i<partCounted;i++){
    ityp = (long)particleSpecies[i * particleSpecies_s];
    opi = output_data + i * output_s0;
    Radoutputip = Radoutputi + i * Radoutputi_s0;
    Angoutputip = Angoutputi + i * Angoutputi_s0;
    Angoutputjp = Angoutputj + i * Angoutputj_s0;
    Angoutputkp = Angoutputk + i * Angoutputk_s0;
    rlistjp = rlistj + i * rlistj_s0;
    alistjp = alistj + i * alistj_s0;
    alistkp = alistk + i * alistk_s0;
    inputp = input_data + (long)PartListOffset[i * PartListOffset_s] * input_s0;

    for (jj = 0; jj < radlistnum[i * radlistnum_s]; ++jj) {
        j = rlistjp[jj * rlistj_s1];
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        offset = (jtyp-1) * num_rad;
        opj = output_data + j * output_s0;
        Radoutputipj = Radoutputip + jj * Radoutputi_s1;

        for (symi=0;symi<num_rad;symi++) { 
            Radoutputips = Radoutputipj + symi * Radoutputi_s2;
            val = inputp[(offset + symi) * input_s1]; 
            for (d = 0; d < 3; d++) {  opi[d * output_s1] -= val * Radoutputips[d * Radoutputi_s3];
                                       opj[d * output_s1] += val * Radoutputips[d * Radoutputi_s3]; }
        } /* loop on symi */

    }  /* loop on jj */

    for (kk = 0; kk < anglistnum[i * anglistnum_s]; ++kk) {
        j = alistjp[kk * alistj_s1];
        k = alistkp[kk * alistk_s1];
        opj = output_data + j * output_s0;
        opk = output_data + k * output_s0;
        Angoutputipj = Angoutputip + kk * Angoutputi_s1;
        Angoutputjpj = Angoutputjp + kk * Angoutputj_s1;
        Angoutputkpj = Angoutputkp + kk * Angoutputk_s1;
        jtyp =  (long)particleSpecies[j * particleSpecies_s];
        ktyp =  (long)particleSpecies[k * particleSpecies_s];
        offset = (jtyp == ktyp) ? (jtyp-1) * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);

        for (symi=0;symi<num_ang;symi++) { 
            Angoutputips = Angoutputipj + symi * Angoutputi_s2;
            Angoutputjps = Angoutputjpj + symi * Angoutputj_s2;
            Angoutputkps = Angoutputkpj + symi * Angoutputk_s2;
            val = inputp[(offset + symi) * input_s1]; 
            for (d = 0; d < 3; d++) {   opi[d * output_s1] += val * Angoutputips[d * Angoutputi_s3]; 
                                        opj[d * output_s1] += val * Angoutputjps[d * Angoutputj_s3]; 
                                        opk[d * output_s1] += val * Angoutputkps[d * Angoutputk_s3]; }
        } /* loop on symi */

    }  /* loop on kk */

  }  /* infinite while loop on i (terminated by break statements above) */
 
} /* if threaded which is an option */

  lua_getfield(L, 1, "output");
  return 1;
}


static const struct luaL_Reg poten_(Energy__) [] = {
  {"Energy_neighborUpdateAll", poten_(Energy_neighborUpdateAll)},
  {"Energy_neighborPairUpdateAll", poten_(Energy_neighborPairUpdateAll)},
  {"Energy_inputUpdateAll", poten_(Energy_inputUpdateAll)},
  {"Energy_updateAll", poten_(Energy_updateAll)},
  {"Energy_updateAllForces", poten_(Energy_updateAllForces)},
  {"Energy_updateAllForceContributions", poten_(Energy_updateAllForceContributions)},
  {"Energy_calculateAllnnForces", poten_(Energy_calculateAllnnForces)},
  {"Energy_calculateAllForces", poten_(Energy_calculateAllForces)},
  {NULL, NULL}
};

static void poten_(Energy_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, poten_(Energy__), "poten");
  lua_pop(L,1);

}

#endif
