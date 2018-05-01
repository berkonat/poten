#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Pair.c"
#else

static int poten_(Pair_updateOutput)(lua_State *L)
{
  THTensor *input  = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *pinput = luaT_getfieldcheckudata(L, 1, "pairInput", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);
  THTensor_(resizeAs)(pinput, input);
  TH_TENSOR_APPLY3(real, input, real, pinput, real, output,  \
                  *pinput_data = *input_data;  \
                   real z2 = (*input_data) * (*input_data); \
                   real z4 = z2 * z2; \
                  *output_data = 1. / (z4 * z4 * z4););
  return 1;
}

static int poten_(Pair_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *input = luaT_getfieldcheckudata(L, 1, "pairInput", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, output);

  if (output->nDimension == 1 || 
      !THTensor_(isContiguous)(gradOutput) || 
      !THTensor_(isContiguous)(output) || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, input, real, output,  \
                    *gradInput_data =  -12. * (*output_data) / (*input_data) ;);
    TH_TENSOR_APPLY2(real, gradInput, real, gradOutput,  \
                    *gradInput_data =  (*gradOutput_data) * (*gradInput_data) ;);
  }
  else
  {
    real* ptr_gradOutput = THTensor_(data)(gradOutput);
    real* ptr_gradInput  = THTensor_(data)(gradInput);
    real* ptr_input      = THTensor_(data)(input);
    real* ptr_output     = THTensor_(data)(output);
    long i;

#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(gradInput); i++)
    {
      ptr_gradInput[i] = ptr_gradOutput[i] * (-12. * ptr_output[i] / ptr_input[i]);
    }
  }
  return 1;
}

static const struct luaL_Reg poten_(Pair__) [] = {
  {"Pair_updateOutput", poten_(Pair_updateOutput)},
  {"Pair_updateGradInput", poten_(Pair_updateGradInput)},
  {NULL, NULL}
};

static void poten_(Pair_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, poten_(Pair__), "poten");
  lua_pop(L,1);

}

#endif
