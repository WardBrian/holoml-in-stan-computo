functions {
  /**
   * Return a beam stop of dimension M1 x M2, with 1 values blocking
   * the lowest frequency portions of an FFT and 0 values elsewhere.
   * The result is an M1 x M2 matrix of 0 values with blocks in the
   * corners set to 1.  The block sizes are upper left (r x r), upper
   * right (r x r-1), lower left (r-1 x r), and lower right (r-1 x
   * r-1).
   * 
   * @param M1 number of rows
   * @param M2 number of cols
   * @param r block dimension
   * @return M1 x M2 matrix with 1 for stopped and 0 for passed values.
   */
  matrix beamstop_gen(int M1, int M2, int r) {
    matrix[M1, M2] B_cal = rep_matrix(0, M1, M2);
    if (r == 0) {
      return B_cal;
    }
    B_cal[1 : r, 1 : r] = rep_matrix(1, r, r);                // upper left
    B_cal[1 : r, M2 - r + 2 : M2] = rep_matrix(1, r, r - 1);  // upper right
    B_cal[M1 - r + 2 : M1, 1 : r] = rep_matrix(1, r - 1, r);  // lower left
    B_cal[M1 - r + 2 : M1, M2 - r + 2 : M2]                   // lower right
      = rep_matrix(1, r - 1, r - 1);
    return B_cal;
  }

  /**
   * Return the matrix corresponding to the fast Fourier transform of
   * Z after it is padded with zeros (to the right and below) to shape
   * N x M.  When N x M is larger than the dimensions of Z, the result
   * is an oversampled two-dimensional discrete Fourier transform.
   *
   * @param Z matrix of values
   * @param N number of rows desired (must be >= rows(Z))
   * @param M number of columns desired (must be >= cols(Z))
   * @return the FFT of Z padded with zeros
   */
  complex_matrix pad_fft2(complex_matrix Z, int N, int M) {
    int r = rows(Z);
    int c = cols(Z);
    complex_matrix[N, M] pad = rep_matrix(0, N, M);
    pad[1 : r, 1 : c] = Z;
    return fft2(pad);
  }

  /**
   * Return the intrinsic conditional autoregressive prior density
   * for the specified image matrix X.  this is 
   *
   * @param X image matrix with values in (0, 1)
   * @param sigma scale of ICAR prior
   * @return ICAR log density for the specified matrix and scale
   */
  real icar_lpdf(matrix X, real sigma) {
    int M = rows(X);
    int N = cols(X);
    return normal_lpdf(to_vector(X[2:M, ]) | to_vector(X[1:M - 1, ]), sigma)
      + normal_lpdf(to_vector(X[ , 2:N]) | to_vector(X[ , 1:N - 1]), sigma);
  }
}
data {
  int<lower=0> N;                    // image dimension
  matrix<lower=0, upper=1>[N, N] R;  // registration image
  int<lower=0, upper=N> d;           // separator
  int<lower=N> M1;                   // padded rows
  int<lower=2 * N + d> M2;           // padded cols
  int<lower=0, upper=M1> r;          // beamstop radius
  real<lower=0> N_p;                 // avg photons per pixel
  array[M1, M2] int<lower=0> Y;      // observed number of photons
  real<lower=0> sigma;               // prior scale
}
transformed data {
  matrix[M1, M2] beamstop = beamstop_gen(M1, M2, r);  // beam stop
  matrix[d, N] Z = rep_matrix(0, d, N);               // separator
  matrix[N, N + d] Z_R = append_col(Z, R);            // separator + ref
}
parameters {
  matrix<lower=0, upper=1>[N, N] X;    // image
}
model {
  X ~ icar(sigma);
  matrix[N, 2 * N + d] X_Z_R = append_col(X, Z_R);
  matrix[M1, M2] V = square(abs(pad_fft2(X_Z_R, M1, M2)));
  matrix[M1, M2] lambda = N_p / mean(V) * V; 
  for (m1 in 1 : M1) {
    for (m2 in 1 : M2) {
      if (!beamstop[m1, m2]) {
        Y[m1, m2] ~ poisson(lambda[m1, m2]);
      }
    }
  }
}
