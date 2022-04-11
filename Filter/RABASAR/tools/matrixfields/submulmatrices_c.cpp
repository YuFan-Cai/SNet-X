#include <stdio.h>
#include <mex.h>
#include <complex>

static void usage()
{
  char str[1024];
  sprintf(str, "usage: z = submulmatrices_c(x, y, optx, opty, optz)\n");
  mexErrMsgTxt(str);
}

template<typename TX, typename TY>
static
void process(int M, int N, int D,
	     std::complex<double>* z,
	     const TX* x, const TY* y,
	     const unsigned char* optx,
	     const unsigned char* opty,
	     const unsigned char* optz)
{
  int i, j, k, l, m;
  int iz, ix, iy;

  switch (optz[0])
    {
      case 'n':
	switch (optx[0])
	  {
	    case 'n':
	    case 'h':
	      switch (opty[0])
		{
		  case 'n': // n:n-n | n:h-n
		  case 'h': // n:n-h | n:h-h
		    for (k = 0; k < D; ++k)
		      for (l = 0; l < D; ++l)
			for (m = 0; m < D; ++m)
			  for (i = 0; i < M; ++i)
			    for (j = 0; j < N; ++j)
			      {
				iz = ((k * D + l) * M + i) * N + j;
				ix = ((m * D + l) * M + i) * N + j;
				iy = ((k * D + m) * M + i) * N + j;
				z[iz] += x[ix] * y[iy];
			      }
		    break;
		  case 'd': // n:n-d | n:h-d
		    for (k = 0; k < D; ++k)
		      for (l = 0; l < D; ++l)
			for (i = 0; i < M; ++i)
			  for (j = 0; j < N; ++j)
			    {
			      iz = ((k * D + l) * M + i) * N + j;
			      ix = ((k * D + l) * M + i) * N + j;
			      iy = ((k * D + k) * M + i) * N + j;
			      z[iz] += x[ix] * y[iy];
			    }
		    break;
		}
	      break;
	    case 'd':
	      switch (opty[0])
		{
		  case 'n': // n:d-n
		  case 'h': // n:d-h
		    for (k = 0; k < D; ++k)
		      for (l = 0; l < D; ++l)
			for (i = 0; i < M; ++i)
			  for (j = 0; j < N; ++j)
			    {
			      iz = ((k * D + l) * M + i) * N + j;
			      ix = ((l * D + l) * M + i) * N + j;
			      iy = ((k * D + l) * M + i) * N + j;
			      z[iz] += x[ix] * y[iy];
			    }
		    break;
		  case 'd': // n:d-d
		    for (k = 0; k < D; ++k)
		      for (i = 0; i < M; ++i)
			for (j = 0; j < N; ++j)
			  {
			    iz = ((k * D + k) * M + i) * N + j;
			    z[iz] += x[iz] * y[iz];
			  }
		    break;
		}
	      break;
	  }
	break;
      case 'd':
	switch (optx[0])
	  {
	    case 'n':
	    case 'h':
	      switch (opty[0])
		{
		  case 'n': // d:n-n | d:h-n
		  case 'h': // d:n-h | d:h-h
		    for (k = 0; k < D; ++k)
		      for (m = 0; m < D; ++m)
			for (i = 0; i < M; ++i)
			  for (j = 0; j < N; ++j)
			    {
			      iz = ((k * D + k) * M + i) * N + j;
			      ix = ((m * D + k) * M + i) * N + j;
			      iy = ((k * D + m) * M + i) * N + j;
			      z[iz] += x[ix] * y[iy];
			    }
		    break;
		  case 'd': // d:n-d | d:h-d
		    for (k = 0; k < D; ++k)
		      for (i = 0; i < M; ++i)
			for (j = 0; j < N; ++j)
			  {
			    iz = ((k * D + k) * M + i) * N + j;
			    z[iz] += x[iz] * y[iz];
			  }
		    break;
		}
	      break;
	    case 'd':
	      switch (opty[0])
		{
		  case 'n': // d:d-n
		  case 'h': // d:d-h
		  case 'd': // d:d-d
		    for (k = 0; k < D; ++k)
		      for (i = 0; i < M; ++i)
			for (j = 0; j < N; ++j)
			  {
			    iz = ((k * D + k) * M + i) * N + j;
			    z[iz] += x[iz] * y[iz];
			  }
		    break;
		}
	      break;
	  }
	break;
      case 'h':
	switch (optx[0])
	  {
	    case 'n':
	    case 'h':
	      switch (opty[0])
		{
		  case 'n': // h:n-n | h:h-n
		  case 'h': // h:n-h | h:h-h
		    for (k = 0; k < D; ++k)
		      for (l = k; l < D; ++l)
			for (m = 0; m < D; ++m)
			  for (i = 0; i < M; ++i)
			    for (j = 0; j < N; ++j)
			      {
				iz = ((k * D + l) * M + i) * N + j;
				ix = ((m * D + l) * M + i) * N + j;
				iy = ((k * D + m) * M + i) * N + j;
				z[iz] += x[ix] * y[iy];
			      }
		    break;
		  case 'd': // h:n-d | h:h-d
		    for (k = 0; k < D; ++k)
		      for (l = k; l < D; ++l)
			for (i = 0; i < M; ++i)
			  for (j = 0; j < N; ++j)
			    {
			      iz = ((k * D + l) * M + i) * N + j;
			      ix = ((k * D + l) * M + i) * N + j;
			      iy = ((k * D + k) * M + i) * N + j;
			      z[iz] += x[ix] * y[iy];
			    }
		    break;
		}
	      break;
	    case 'd':
	      switch (opty[0])
		{
		  case 'n': // h:d-n
		  case 'h': // h:d-h
		    for (k = 0; k < D; ++k)
		      for (l = k; l < D; ++l)
			for (i = 0; i < M; ++i)
			  for (j = 0; j < N; ++j)
			    {
			      iz = ((k * D + l) * M + i) * N + j;
			      ix = ((l * D + l) * M + i) * N + j;
			      iy = ((k * D + l) * M + i) * N + j;
			      z[iz] += x[ix] * y[iy];
			    }
		    break;
		  case 'd': // h:d-d
		    for (k = 0; k < D; ++k)
		      for (i = 0; i < M; ++i)
			for (j = 0; j < N; ++j)
			  {
			    iz = ((k * D + k) * M + i) * N + j;
			    z[iz] += x[iz] * y[iz];
			  }
		    return;
		}
	      break;
	  }
	for (k = 0; k < D; ++k)
	  for (l = k+1; l < D; ++l)
	    for (i = 0; i < M; ++i)
	      for (j = 0; j < N; ++j)
		{
		  iz = ((k * D + l) * M + i) * N + j;
		  ix = ((l * D + k) * M + i) * N + j;
		  z[ix] = conj(z[iz]);
		}
	break;
    }
}


void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  std::complex<double>* z;
  const void* x;
  const void* y;
  const mwSize* s;
  const unsigned char*   optx;
  const unsigned char*   opty;
  const unsigned char*   optz;
  int S, M, N, D;

  if (nrhs != 5 || nlhs != 1)
    {
      usage();
      return;
    }
  S = mxGetNumberOfDimensions(prhs[0]);
  s = mxGetDimensions(prhs[0]);
  x = mxGetData(prhs[0]);
  y = mxGetData(prhs[1]);
  optx = (unsigned char*) mxGetChars(prhs[2]);
  opty = (unsigned char*) mxGetChars(prhs[3]);
  optz = (unsigned char*) mxGetChars(prhs[4]);
  plhs[0] = mxCreateNumericArray(S, s, mxDOUBLE_CLASS, mxCOMPLEX);
  z = (std::complex<double>*) mxGetData(plhs[0]);

  M = s[0];
  N = s[1];
  if (S == 2)
    D = 1;
  else
    D = s[2];
  if (mxIsComplex(prhs[0]) && mxIsDouble(prhs[0]))
    {
      if (mxIsComplex(prhs[1]) && mxIsDouble(prhs[1]))
	{
	  process<std::complex<double>, std::complex<double>>(M, N, D, z,
							      (const std::complex<double>*) x,
							      (const std::complex<double>*) y,
							      optx, opty, optz);
	  return;
	}
      if (!mxIsComplex(prhs[1]) && mxIsDouble(prhs[1]))
	{
	  process<std::complex<double>, double>(M, N, D, z,
						(const std::complex<double>*) x,
						(const double*) y,
						optx, opty, optz);
	  return;
	}
    }
  if (!mxIsComplex(prhs[0]) && mxIsDouble(prhs[0]))
    {
      if (mxIsComplex(prhs[1]) && mxIsDouble(prhs[1]))
	{
	  process<double, std::complex<double>>(M, N, D, z,
						(const double*) x,
						(const std::complex<double>*) y,
						optx, opty, optz);
	  return;
	}
      if (!mxIsComplex(prhs[1]) && mxIsDouble(prhs[1]))
	{
	  process<double, double>(M, N, D, z,
				  (const double*) x,
				  (const double*) y,
				  optx, opty, optz);
	  return;
	}
    }
  return;
}
