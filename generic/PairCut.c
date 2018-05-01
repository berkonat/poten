#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/PairCut.c"
#else

static int poten_(PairCut_updateOutput)(lua_State *L)
{
  THTensor *input  = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *pinput = luaT_getfieldcheckudata(L, 1, "pairInput", torch_Tensor);
  THTensor *cutoff = luaT_getfieldcheckudata(L, 1, "pairCutoff", torch_Tensor);
  THTensor *fcutij = luaT_getfieldcheckudata(L, 1, "pairFcutij", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);
  THTensor_(resizeAs)(pinput, input);
  THTensor_(resizeAs)(cutoff, input);
  THTensor_(resizeAs)(fcutij, input);

  TH_TENSOR_APPLY2(real, input, real, pinput,  \
                  *pinput_data = *input_data;);
#if defined(TH_REAL_IS_FLOAT) 
  TH_TENSOR_APPLY3(real, cutoff, real, input, real, fcutij,  \
                   real pi=3.1415926535897932384626433832795028841971694; \
                  *fcutij_data = (*input_data < *cutoff_data) ? 0.5 * (cosf(pi * *input_data / *cutoff_data ) + 1.0) : 0.0 ;);
#else
  TH_TENSOR_APPLY3(real, cutoff, real, input, real, fcutij,  \
                   real pi=3.1415926535897932384626433832795028841971694; \
                  *fcutij_data = (*input_data < *cutoff_data) ? 0.5 * (cos(pi * *input_data / *cutoff_data ) + 1.0) : 0.0 ;);
#endif
  TH_TENSOR_APPLY3(real, input, real, fcutij, real, output,  \
                   real z2 = (*input_data) * (*input_data); \
                   real z4 = z2 * z2; \
                  *output_data = (1. / (z4 * z4 * z4)) * *fcutij_data;);
  return 1;
}

static int poten_(PairCut_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *input = luaT_getfieldcheckudata(L, 1, "pairInput", torch_Tensor);
  THTensor *cutoff = luaT_getfieldcheckudata(L, 1, "pairCutoff", torch_Tensor);
  THTensor *fcutij = luaT_getfieldcheckudata(L, 1, "pairFcutij", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, output);

  if (output->nDimension == 1 || 
      !THTensor_(isContiguous)(gradOutput) || 
      !THTensor_(isContiguous)(output) || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(cutoff) || 
      !THTensor_(isContiguous)(fcutij) || 
      !THTensor_(isContiguous)(gradInput))
  {
    // Temperory Tensor
    THTensor * fgrad = THTensor_(new)();
    THTensor_(resizeAs)(fgrad, output);
    
    TH_TENSOR_APPLY3(real, gradInput, real, input, real, output,  \
                    *gradInput_data =  -12. * (*output_data) / (*input_data) ;);
    TH_TENSOR_APPLY2(real, gradInput, real, fcutij,  \
                    *gradInput_data =  *gradInput_data * *fcutij_data ;);
#if defined(TH_REAL_IS_FLOAT)
    TH_TENSOR_APPLY3(real, fgrad, real, input, real, cutoff,  \
                     real pi=3.1415926535897932384626433832795028841971694; \
                    *fgrad_data = (*input_data < *cutoff_data) ? -(0.5 * pi/ *cutoff_data ) * sinf(pi * *input_data / *cutoff_data ) : 0.0 ;);
#else 
    TH_TENSOR_APPLY3(real, fgrad, real, input, real, cutoff,  \
                     real pi=3.1415926535897932384626433832795028841971694; \
                    *fgrad_data = (*input_data < *cutoff_data) ? -(0.5 * pi/ *cutoff_data ) *  sin(pi * *input_data / *cutoff_data ) : 0.0 ;);
#endif 
    TH_TENSOR_APPLY3(real, gradInput, real, output, real, fgrad,  \
                    *gradInput_data =  *fgrad_data * (*output_data) + (*gradInput_data) ;);
    TH_TENSOR_APPLY2(real, gradInput, real, gradOutput,  \
                    *gradInput_data =  (*gradOutput_data) * (*gradInput_data) ;);
    // Cleaning
    THTensor_(free)(fgrad);
  }
  else
  {
    real* ptr_gradOutput = THTensor_(data)(gradOutput);
    real* ptr_gradInput  = THTensor_(data)(gradInput);
    real* ptr_cutoff     = THTensor_(data)(cutoff);
    real* ptr_fcutij     = THTensor_(data)(fcutij);
    real* ptr_input      = THTensor_(data)(input);
    real* ptr_output     = THTensor_(data)(output);
    long i;
    real pi=3.1415926535897932384626433832795028841971694;

#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(gradInput); i++)
    {
#if defined(TH_REAL_IS_FLOAT)
      real fcutij_gradi = (ptr_input[i] < ptr_cutoff[i]) ? -(0.5 * pi/ptr_cutoff[i]) * sinf(pi * ptr_input[i]/ptr_cutoff[i]) : 0.0 ; 
#else
      real fcutij_gradi = (ptr_input[i] < ptr_cutoff[i]) ? -(0.5 * pi/ptr_cutoff[i]) *  sin(pi * ptr_input[i]/ptr_cutoff[i]) : 0.0 ; 
#endif
      ptr_gradInput[i] = ptr_gradOutput[i] * ( (-12. * ptr_output[i] / ptr_input[i]) * ptr_fcutij[i] + ptr_output[i] * fcutij_gradi);
    }
  }
  return 1;
}

static const struct luaL_Reg poten_(PairCut__) [] = {
  {"PairCut_updateOutput", poten_(PairCut_updateOutput)},
  {"PairCut_updateGradInput", poten_(PairCut_updateGradInput)},
  {NULL, NULL}
};

static void poten_(PairCut_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, poten_(PairCut__), "poten");
  lua_pop(L,1);

}

#endif
