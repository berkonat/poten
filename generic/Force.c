#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Force.c"
#else

static real poten_(Force_c_vnorm)(lua_State *L, real *v){
#if defined(TH_REAL_IS_FLOAT)
  return sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
#else
  return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
#endif
}

static real poten_(Force_c_fcut)(lua_State *L, real *cutoff, real *dij){
  real pi=3.1415926535897932384626433832795028841971694;
  if (*dij < *cutoff) {
#if defined(TH_REAL_IS_FLOAT)
    return 0.5*(cosf(pi * *dij / *cutoff )+1.0); 
#else
    return 0.5*(cos(pi * *dij / *cutoff )+1.0); 
#endif
  } else {
    return 0.0;
  }
}

static void poten_(Force_c_fcutdr)(lua_State *L, real *grad, real *cutoff, real *dij, real *Rij){
  int i;
  real value;
  real pi=3.1415926535897932384626433832795028841971694;
  if (*dij < *cutoff) {
#if defined(TH_REAL_IS_FLOAT)
    value = -(0.5*pi/ *cutoff )*sinf(pi * *dij/ *cutoff ); 
#else
    value = -(0.5*pi/ *cutoff )*sin(pi * *dij/ *cutoff ); 
#endif
    for(i=0;i<3;i++) grad[i] = value * Rij[i] / *dij;
  } else {
    for(i=0;i<3;i++) grad[i] = 0.0; 
  }
}

static real poten_(Force_c_costheta)(lua_State *L, real *dij, real *dik, real *Rij, real *Rik){
  real value=0.0;
  int i;
  for(i=0;i<3;i++) value += Rij[i]*Rik[i];
  return value * (1.0/( *dij * *dik));
}

static void poten_(Force_c_acosthetadr)(lua_State *L, real *gradi, real *gradj, real *gradk, real *dij, real *dik, real *Rij, real *Rik){
  real value=0.0,normval;
  real norm1,norm2;
  int i;
  norm1=1.0/(*dij * *dij);
  norm2=1.0/(*dik * *dik);
  normval=1.0/(*dij * *dik);
  for(i=0;i<3;i++) value += Rij[i]*Rik[i];
  for(i=0;i<3;i++){
     gradi[i] = ( ( Rij[i] + Rik[i] ) - value*( Rij[i]*norm1 + Rik[i]*norm2 ) )*normval;
     gradj[i] = ( value*Rij[i]*norm1 - Rik[i] ) * normval;
     gradk[i] = ( value*Rik[i]*norm2 - Rij[i] ) * normval;
  } 
}

static real poten_(Force_c_radsymf)(lua_State *L, real *rad_eta, real *cutoff, real *Rij){
  real dij;
  dij = poten_(Force_c_vnorm)(L, Rij);
#if defined(TH_REAL_IS_FLOAT)
  return expf(- *rad_eta *dij*dij) * poten_(Force_c_fcut)(L, cutoff, &dij); 
#else
  return exp(- *rad_eta *dij*dij) * poten_(Force_c_fcut)(L, cutoff, &dij); 
#endif
}

static void poten_(Force_c_radsymfdr)(lua_State *L, real *ff, real *input, long *offset, long *num_rad, real *rad_eta, real *cutoff, real *Rij){
  int i,symi;
  real dij,dij2,fcut_grad[3],value,vexp;
  dij = poten_(Force_c_vnorm)(L, Rij);
  value = poten_(Force_c_fcut)(L, cutoff, &dij);
  poten_(Force_c_fcutdr)(L, fcut_grad, cutoff, &dij, Rij); 
  dij2 = dij * dij;
  for (symi=0;symi< *num_rad;symi++) { 
#if defined(TH_REAL_IS_FLOAT)
      vexp = input[ *offset + symi] * expf(-rad_eta[symi] * dij2); 
#else
      vexp = input[ *offset + symi] * exp(-rad_eta[symi] * dij2); 
#endif
      for(i=0;i<3;i++){
         //ff[i] += input[(*offset)+symi] * ((-2.0*rad_eta[symi]*Rij[i]*exp(-rad_eta[symi]*dij2)*value) + (exp(-rad_eta[symi]*dij2)*fcut_grad[i])); 
         ff[i] += vexp * ( -2.0*rad_eta[symi]*Rij[i]*value + fcut_grad[i] ); 
      }
  }
}

static real poten_(Force_c_angsymf)(lua_State *L, real ang_eta, real lambda, real zeta, real *cutoff, real *Rij, real *Rik, real *Rjk){
  real dij,djk,dik;
  real fcutall,power1,power2;
  dij = poten_(Force_c_vnorm)(L, Rij);
  djk = poten_(Force_c_vnorm)(L, Rjk);
  dik = poten_(Force_c_vnorm)(L, Rik);
  fcutall = poten_(Force_c_fcut)(L, cutoff, &dik) * 
            poten_(Force_c_fcut)(L, cutoff, &dij) *
            poten_(Force_c_fcut)(L, cutoff, &djk);
#if defined(TH_REAL_IS_FLOAT)
  power1 = powf((1.0+lambda* poten_(Force_c_costheta)(L, &dij, &dik, Rij, Rik) ),zeta);
  power2 = powf(2.0,(1.0-zeta));
  return expf(-ang_eta*(dik*dik+dij*dij+djk*djk))*fcutall*power1*power2;
#else
  power1 = pow((1.0+lambda* poten_(Force_c_costheta)(L, &dij, &dik, Rij, Rik) ),zeta);
  power2 = pow(2.0,(1.0-zeta));
  return exp(-ang_eta*(dik*dik+dij*dij+djk*djk))*fcutall*power1*power2;
#endif
}

static void poten_(Force_c_angsymfdr)(lua_State *L, real *ffi, real *ffj, real *ffk, real *input, long *offset, long *num_ang, real *ang_eta, real *lambda, real *zeta, real *cutoff, real *Rij, real *Rik, real *Rjk){
  int i,symi;
  real dij,djk,dik;
  real thetagradi[3],thetagradj[3],thetagradk[3];
  real expon,powksi,power2,powksi1,value,value1;
  real fcut_gradi_ij[3],fcut_gradi_ik[3];
  real fcut_gradj_ij[3],fcut_gradj_jk[3];
  real fcut_gradk_ik[3],fcut_gradk_jk[3];
  real fc_ij,fc_ik,fc_jk;
  real costhetaval;
  dij = poten_(Force_c_vnorm)(L, Rij);
  djk = poten_(Force_c_vnorm)(L, Rjk);
  dik = poten_(Force_c_vnorm)(L, Rik);
  costhetaval = poten_(Force_c_costheta)(L, &dij, &dik, Rij, Rik); 
  poten_(Force_c_acosthetadr)(L, thetagradi, thetagradj, thetagradk, &dij, &dik, Rij, Rik);
  poten_(Force_c_fcutdr)(L, fcut_gradi_ij, cutoff, &dij, Rij); 
  poten_(Force_c_fcutdr)(L, fcut_gradi_ik, cutoff, &dik, Rik); 
  poten_(Force_c_fcutdr)(L, fcut_gradj_jk, cutoff, &djk, Rjk); 
  poten_(Force_c_fcutdr)(L, fcut_gradj_ij, cutoff, &dij, Rij); 
  poten_(Force_c_fcutdr)(L, fcut_gradk_ik, cutoff, &dik, Rik); 
  poten_(Force_c_fcutdr)(L, fcut_gradk_jk, cutoff, &djk, Rjk); 
  fc_ij = poten_(Force_c_fcut)(L, cutoff, &dij);
  fc_ik = poten_(Force_c_fcut)(L, cutoff, &dik);
  fc_jk = poten_(Force_c_fcut)(L, cutoff, &djk);
  for (symi=0;symi< *num_ang;symi++) { 
#if defined(TH_REAL_IS_FLOAT)
      expon   = expf(-ang_eta[symi]*(dik*dik+dij*dij+djk*djk));
      powksi  = powf(1.0+lambda[symi]*costhetaval,(real)zeta[symi]);
      power2  = powf(2.0,(real)(1.0-zeta[symi]));
      powksi1 = powf(1.0+lambda[symi]*costhetaval,(real)(zeta[symi]-1.0))*zeta[symi]*lambda[symi]*expon*fc_ij*fc_ik*fc_jk;
#else
      expon   = exp(-ang_eta[symi]*(dik*dik+dij*dij+djk*djk));
      powksi  = pow(1.0+lambda[symi]*costhetaval,(real)zeta[symi]);
      power2  = pow(2.0,(real)(1.0-zeta[symi]));
      powksi1 = pow(1.0+lambda[symi]*costhetaval,(real)(zeta[symi]-1.0))*zeta[symi]*lambda[symi]*expon*fc_ij*fc_ik*fc_jk;
#endif
      value  = input[ *offset + symi] * power2; 
      value1 = -ang_eta[symi]*2.0*expon*fc_ij*fc_ik*fc_jk;
      for(i=0;i<3;i++){
          ffi[i] -=  value * ( ( thetagradi[i]*powksi1 ) +  
                           powksi * ( ( ( Rij[i] + Rik[i] ) * value1 ) + 
                                      ( expon * fc_jk * ( fcut_gradi_ij[i] * fc_ik + fc_ij * fcut_gradi_ik[i] ) )
                                    )
                             );
          ffj[i] -=  value * ( ( thetagradj[i]*powksi1 ) +  
                           powksi * ( ( ( -Rij[i] + Rjk[i] ) * value1 ) + 
                                      ( expon * fc_ik * ( -fcut_gradj_ij[i] * fc_jk + fc_ij * fcut_gradj_jk[i] ) )
                                    )
                             );
          ffk[i] -=  value * ( ( thetagradk[i]*powksi1 ) +  
                           powksi * ( ( ( -Rik[i] - Rjk[i] ) * value1 ) + 
                                      ( expon * fc_ij * ( -fcut_gradk_ik[i] * fc_jk + fc_ik * -fcut_gradk_jk[i] ) )
                                    )
                             );
      }
  }
}

static int poten_(Force_updateOutput)(lua_State *L)
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
  
  printf("numParts: %lf\n",partCounted);
  printf("numsym: %lf\n",num_syms);
  THTensor_(resize2d)(output, partCounted, 3);
  THTensor_(zero)(output);
  real* output_data = THTensor_(data)(output);
  long  output_s0 = THTensor_(stride)(output, 0);
  long  output_s1 = THTensor_(stride)(output, 1);
  real* outputpi;
  real* outputpj;
  real* outputpk;

  printf("Looks good\n");
  real* input_data  = THTensor_(data)(input);
  printf("input: %lf\n",input_data[0]);
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
  real Rij[3],Rji[3],Rik[3],Rjk[3];
  real fcutall, fcutik, fcutij, fcutjk;
  real fcut_gradi_ij, fcut_gradi_ik, fcut_gradj_jk;
  real thetagradi[3],thetagradj[3],thetagradk[3];
  real costheta, costhetaVALUE;
  real power1, power2, vexp1, vexp2;
  real powksi1, expon, angexp;
  real RijRikV, dijdijV, dijdikV, dikdikV;
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
    ityp =  (long)particleSpecies[i * particleSpecies_s]-1;
    outputpi = output_data + i * output_s0;
    inputp = input_data + ((long)PartListOffset[i * PartListOffset_s]-1) * input_s0;
    Rijpi = Rij_list + i * Rij_list_s0;

    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = (long)partNLp[jj * neighListOfParts_s1];
      Rsqij = 0.0;
      Rijpj = Rijpi + jj * Rij_list_s1;
      for (d = 0; d < 3; ++d) {
        Rij[d] =  Rijpj[d * Rij_list_s2]; 
        Rji[d] = -Rij[d];
        Rsqij +=  Rij[d] * Rij[d]; 
      } /* End of Rij[k] loop */

      if (Rsqij < cutsq) {
        for (d = 0; d < 3; d++)  ffi[d] = 0.0;
        jtyp =  (long)particleSpecies[j * particleSpecies_s]-1;
        offset = jtyp * num_rad;
        outputpj = output_data + j * output_s0;

#if defined(TH_REAL_IS_FLOAT)
        dij = sqrtf(Rsqij);
        dijdijV = 1.0/(dij * dij);
        fcutij = (dij < cutoff) ? 0.5 * (cosf(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        fcut_gradi_ij = (dij < cutoff) ? -(0.5 * pi/cutoff ) * sinf(pi * dij/cutoff ) / dij : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            vexp2 = inputp[(offset + symi) * input_s1] * expf( -rad_eta[rad_eta_s * symi] * dijdijV); 
            vexp1 = -2.0 * rad_eta[rad_eta_s * symi] * fcutij * vexp2; 
            for (d = 0; d < 3; d++) ffi[d] += ( vexp1 + vexp2 * fcut_gradi_ij ) * Rij[d]; 
        }
#else
        dij = sqrt(Rsqij);
        dijdijV = 1.0/(dij * dij);
        fcutij = (dij < cutoff) ? 0.5 * (cos(pi * dij / cutoff ) + 1.0) : 0.0 ; 
        fcut_gradi_ij = (dij < cutoff) ? -(0.5 * pi/cutoff ) * sin(pi * dij/cutoff ) / dij : 0.0 ; 
        for (symi=0;symi<num_rad;symi++) { 
            vexp2 = inputp[(offset + symi) * input_s1] * exp( -rad_eta[rad_eta_s * symi] * dijdijV); 
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
        
          //if (Rsqik < cutsq && Rsqjk < cutsq && Rsqik > 0.000000001 && Rsqjk > 0.000000001) {
          if (Rsqik < cutsq && Rsqik > 0.001 && Rsqjk > 0.001) {
             outputpk = output_data + k * output_s0;
             ktyp =  (long)particleSpecies[k * particleSpecies_s]-1;
             offset = (jtyp == ktyp) ? jtyp * num_ang + num_typ * num_rad : num_typ * (num_ang + num_rad);
             RijRikV = 0.0;
             for (d = 0; d < 3; d++) {   ffi[d] = 0.0; ffj[d] = 0.0; ffk[d] = 0.0; RijRikV += Rij[d]*Rik[d]; }

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

             dikdikV = 1.0/(dik * dik);
             dijdikV = 1.0/(dij * dik);
             for(d=0;d<3;d++){
                thetagradi[d] = ( ( Rij[d] + Rik[d] ) - RijRikV * ( Rij[d] * dijdijV + Rik[d] * dikdikV ) ) * dijdikV;
                thetagradj[d] = ( RijRikV * Rij[d] * dijdijV - Rik[d] ) * dijdikV;
                thetagradk[d] = ( RijRikV * Rik[d] * dikdikV - Rij[d] ) * dijdikV;
             } 

             fcutall = fcutik * fcutij * fcutjk;
             dall = dik*dik + dij*dij + djk*djk ;
             costhetaVALUE = RijRikV * dijdikV;

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

static const struct luaL_Reg poten_(Force__) [] = {
  {"Force_updateOutput", poten_(Force_updateOutput)},
  {NULL, NULL}
};

static void poten_(Force_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, poten_(Force__), "poten");
  lua_pop(L,1);

}

#endif
