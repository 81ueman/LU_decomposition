#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <stdalign.h>
#include <immintrin.h>

#define N 4800
#define bsize 48
alignas(32) double a[N][N];
double b[N];
double y[N];
double x[N];
double orig[N][N];

void show_debug();
void result_debug();

double elapsed(struct timeval start, struct timeval end){
    return end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1e6;
}

void simpleLU(int ind){
        for(int i=ind; i<ind+bsize; i++){
                double fac = 1.0 / a[i][i];
                for(int j=i+1; j<ind+bsize; j++){
                    a[j][i] *= fac;
                }
                for(int j=i+1; j<ind+bsize; j++){
                    for(int k = i+1; k<ind+bsize; k++){
                        a[j][k] -= a[j][i] * a[i][k];
                    }
                }
        }
}
//L21 = A_21 U_11^-1
// xU = A_21 solve x
// ind for left-upper of U
void Uinv(int ind){
#pragma omp parallel for collapse(1)
    for(int row = ind+bsize; row<N; row++){
        for(int col=ind; col<ind + bsize; col++){
            for(int j=ind; j<col; j++){
                a[row][col] -= a[row][j] * a[j][col];
            }
            a[row][col]  /= a[col][col];
        }
    }
}

// U_12 = L_11^-1 A_12
// solve L_11 U_12 = A_12 respectively for U
void Linv(int ind){
#pragma omp parallel for collapse(1)
    for(int col=ind+bsize; col<N; col++){
        for(int row=ind; row<ind + bsize; row++){
            for(int j=ind; j<row; j++){
                a[row][col] -= a[row][j] * a[j][col];
            }
        }
    }
}

void show_256(__m256d r, char *s){
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp,r);
        printf("%s %lf %lf %lf %lf", s, tmp[0], tmp[1], tmp[2], tmp[3]);


}
void A22(int ind){
#pragma omp parallel for collapse(2)
    for(int i = ind+bsize; i<N; i+=4){
        for(int j= ind + bsize; j<N; j+=4){
            __m256d s0,s1,s2,s3;
            s0 = _mm256_xor_pd(s0,s0);
            s1 = _mm256_xor_pd(s0,s0);
            s2 = _mm256_xor_pd(s0,s0);
            s3 = _mm256_xor_pd(s0,s0);
            for(int k= ind; k<ind+bsize; k++){
                alignas(32) double tmp[4],u[4];
                tmp[0] = a[i][k];
                tmp[1] = a[i+1][k];
                tmp[2] = a[i+2][k];
                tmp[3] = a[i+3][k];
                __m256d l,r;
                l = _mm256_load_pd(tmp);
                r = _mm256_load_pd(a[k] + j);
                s0 = _mm256_fmadd_pd(l,r,s0);
                l = _mm256_permute_pd(l, 0b0101);
                s1 = _mm256_fmadd_pd(l,r,s1);
                l = _mm256_permute4x64_pd(l, 0b01001110);
                s2 = _mm256_fmadd_pd(l,r,s2);
                l = _mm256_permute_pd(l, 0b0101);
                s3 = _mm256_fmadd_pd(l,r,s3);
            }
        __m256d t0,t1,t2,t3;
            t0 = _mm256_blend_pd(s0,s1, 0b0101);
            t1 = _mm256_blend_pd(s0,s1, 0b1010);
            t2 = _mm256_blend_pd(s2,s3, 0b0101);
            t3 = _mm256_blend_pd(s2,s3,0b1010);
            __m256d c0,c1,c2,c3;
            c3 = _mm256_blend_pd(t0,t3, 0b0011);
            c2 = _mm256_blend_pd(t1,t2, 0b0011);
            c1 = _mm256_blend_pd(t0,t3, 0b1100);
            c0 = _mm256_blend_pd(t1,t2, 0b1100);
            __m256d a0 = _mm256_load_pd(a[i+0]+j);
            a0 = _mm256_sub_pd(a0, c0);
            _mm256_store_pd(a[i+0]+j,a0);

            __m256d a1 = _mm256_load_pd(a[i+1]+j);
            a1 = _mm256_sub_pd(a1, c1);
            _mm256_store_pd(a[i+1]+j,a1);

            __m256d a2 = _mm256_load_pd(a[i+2]+j);
            a2 = _mm256_sub_pd(a2, c2);
            _mm256_store_pd(a[i+2]+j,a2);

            __m256d a3 = _mm256_load_pd(a[i+3]+j);
            a3 = _mm256_sub_pd(a3, c3);
            _mm256_store_pd(a[i+3]+j,a3);

        }
    }

}




void LU(){
    for(int i=0; i<N; i+=bsize){ // loop N/ block times 
        simpleLU(i);   // block^3 * 2 / 3          ---> N * block^ * 2 /3
        Linv(i);       // block^2 * (N - (i+1)*block)---> 2 block^1 * N^2 - block^1 * N^2
        Uinv(i);       // block^2 * (N - (i+1)*block)---> 2 block^1 * N^2 - block^1 * N^2
        A22(i);        // 2block * (N - (i+1)*block)^2-->2  N^3 - 2block^1  N^2 + block^1 N^2
                       // sum 2 N^3 + block N^2 + block^2 N * 2 /3
    }
}

void solve(){
    //L
    for(int i=0; i<N; i++){
        y[i] = b[i];
        for(int j=0; j<i; j++){
            y[i] -= y[j] * a[i][j];
        }
    }
    //U
    for(int i=N-1; i>=0; i--){
        x[i] = y[i];
        for(int j=i+1; j<N; j++){
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
}
int main(){
    srand((unsigned int) time(NULL));
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            a[i][j] = (double)rand() / RAND_MAX;
            orig[i][j] = a[i][j];
        }
        b[i] = (double)rand() / RAND_MAX;
    }

    printf("\n");
    show_debug();

    struct timeval start;
    gettimeofday(&start,NULL);

    LU();

    struct timeval end;
    gettimeofday(&end,NULL);


    solve();
    result_debug();
    double time = elapsed(start, end);
    printf("elapsed time:%f\n", time);
    double flops = ((long long)  2 * N*N*N + (long long)bsize* N * N +(long long) bsize* bsize * N * 2 / 3) / time / 1e9;
    printf("flops:%lf\n", flops);


}
void show_debug(){
#ifdef DEBUG
    printf("A\n");
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            printf("%lf ", a[i][j]);
        }
        printf("\n");
    }
    printf("b\n");
    for(int i=0; i<N; i++){
        printf("%lf " ,b[i]);
    }
    printf("\n");
    #endif
}
void result_debug(){
    #ifdef DEBUG

        printf("LU\n");
        for(int i=0; i<N; i++){
                for(int j=0; j<N; j++){
                        double sum = 0.0;
                        for(int k=0; k<N; k++){
                                sum += (i>k ? a[i][k] : i==k ? 1.0 : 0.0)  * (k<=j ? a[k][j] : 0.0);
                        }
                        printf("%.6lf ", sum - orig[i][j]);
                }
                printf("\n");
        }
    printf("L\n");
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(i==j){
                printf("1.0 ");
            }else if(i>j){
                printf("%lf ", a[i][j]);
            }else{
                printf("0.0 ");
            }
        }
        printf("\n");
    }
    printf("\n");
    printf("U\n");
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(i<=j){
                printf("%lf ", a[i][j]);
            }else{
                printf("0.0 ");
            }
        }
     printf("\n");
    }
    printf("y\n");
    for(int i=0; i<N; i++){
        printf("%lf ", y[i]);
    }
    printf("\n");

    printf("x\n");
    for(int i=0; i<N; i++){
        printf("%lf ", x[i]);
    }
    printf("\n");
    printf("diff\n");
    for(int i=0; i<N; i++){
            double diff  = b[i];
            for(int j=0; j<N; j++){
                    diff -= orig[i][j] * x[j];
            }
            printf("%lf ", diff);
    }
    printf("\n");

    #endif

}
                              
