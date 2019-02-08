//
// Created by Pascal on 08.06.2018.
//

#ifndef BLAS_HLS_BLAS_H
#define BLAS_HLS_BLAS_H

#ifndef WIDTH
#define WIDTH   16
#endif

#define BLAS_FUNCTION(func)   fblas_##func

#ifdef __cplusplus
extern "C" {
#endif

	void prepareFPGAForSingle();
	void prepareFPGAForDouble();

	// Level 1
	void BLAS_FUNCTION(saxpy)(int N, float ALPHA, const float *X, int INCX, float *Y, int INCY);
	void BLAS_FUNCTION(daxpy)(int N, double ALPHA, const double *X, int INCX, double *Y, int INCY);
	float BLAS_FUNCTION(sdot)(int N, const float *X, int INCX, const float *Y, int INCY);
	double BLAS_FUNCTION(ddot)(int N, const double *X, int INCX, const double *Y, int INCY);
	float BLAS_FUNCTION(snrm2)(int N, const float *X, int INCX);
	double BLAS_FUNCTION(dnrm2)(int N, const double *X, int INCX);

	// Level 2
	void BLAS_FUNCTION(sgemv)(char TRANS, int M, int N, float ALPHA, const float *A, int LDA, const float *X, int INCX, float BETA, float *Y, int INCY);
	void BLAS_FUNCTION(dgemv)(char TRANS, int M, int N, double ALPHA, const double *A, int LDA, const double *X, int INCX, double BETA, double *Y, int INCY);

#ifdef __cplusplus
}
#endif

#endif //BLAS_HLS_BLAS_H
