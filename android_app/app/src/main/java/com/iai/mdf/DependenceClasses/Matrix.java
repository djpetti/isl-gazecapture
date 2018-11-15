package com.iai.mdf.DependenceClasses;

/**
 * Created by mou on 11/8/18.
 */


import android.util.Log;

/******************************************************************************
 *  Compilation:  javac Matrix.java
 *  Execution:    java Matrix
 *
 *  A bare-bones immutable data type for M-by-N matrices.
 *
 ******************************************************************************/



public class Matrix {
    private final int M;             // number of rows
    private final int N;             // number of columns
    private final double[][] data;   // M-by-N array

    // create M-by-N matrix of 0's
    public Matrix(int M, int N) {
        this.M = M;
        this.N = N;
        data = new double[M][N];
    }

    // create matrix based on 2d array
    public Matrix(double[][] data) {
        M = data.length;
        N = data[0].length;
        this.data = new double[M][N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                this.data[i][j] = data[i][j];
    }

    // copy constructor
    private Matrix(Matrix A) { this(A.data); }


    // element access
    public double get(int row, int col){
        return data[row][col];
    }


    // create and return a random M-by-N matrix with values between 0 and 1
    public static Matrix random(int M, int N) {
        Matrix A = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[i][j] = Math.random();
        return A;
    }

    // create and return the N-by-N identity matrix
    public static Matrix identity(int N) {
        Matrix I = new Matrix(N, N);
        for (int i = 0; i < N; i++)
            I.data[i][i] = 1;
        return I;
    }

    // swap rows i and j
    private void swap(int i, int j) {
        double[] temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }

    // create and return the transpose of the invoking matrix
    public Matrix transpose() {
        Matrix A = new Matrix(N, M);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[j][i] = this.data[i][j];
        return A;
    }

    // return C = A + B
    public Matrix plus(Matrix B) {
        Matrix A = this;
        if (B.M != A.M || B.N != A.N) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C.data[i][j] = A.data[i][j] + B.data[i][j];
        return C;
    }


    // return C = A - B
    public Matrix minus(Matrix B) {
        Matrix A = this;
        if (B.M != A.M || B.N != A.N) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C.data[i][j] = A.data[i][j] - B.data[i][j];
        return C;
    }

    // does A = B exactly?
    public boolean eq(Matrix B) {
        Matrix A = this;
        if (B.M != A.M || B.N != A.N) throw new RuntimeException("Illegal matrix dimensions.");
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (A.data[i][j] != B.data[i][j]) return false;
        return true;
    }

    // return C = A * B
    public Matrix times(Matrix B) {
        Matrix A = this;
        if (A.N != B.M) throw new RuntimeException("Illegal matrix dimensions.");
        Matrix C = new Matrix(A.M, B.N);
        for (int i = 0; i < C.M; i++)
            for (int j = 0; j < C.N; j++)
                for (int k = 0; k < A.N; k++)
                    C.data[i][j] += (A.data[i][k] * B.data[k][j]);
        return C;
    }


    public float[] timesArray(float[] rArr) {
        if (rArr.length != M)
            throw new RuntimeException("Illegal matrix dimensions.");
        double[][] formattedArray = new double[M][1];
        for (int i=0; i<M; i++){
            formattedArray[i][0] = rArr[i];
        }
        Matrix rMat = new Matrix(formattedArray);
        Matrix res = this.times(rMat);
        return new float[] {   (float)res.get(0,0), (float)res.get(1,0)};
    }

    public Matrix inverse2x2(){
        double[][] inv = new double[2][2];
        double det = data[0][0] * data[1][1] - data[0][1] * data[1][0];
        inv[0][0] = data[1][1] / det;
        inv[0][1] = - data[0][1] / det;
        inv[1][0] = - data[1][0] / det;
        inv[1][1] = data[0][0] / det;
        return new Matrix(inv);
    }

    public Matrix inverse3x3(){
        double[][] inv = new double[3][3];
        double det = data[0][0] * data[1][1] * data[2][2]
                    + data[0][1] * data[1][2] * data[2][0]
                    + data[0][2] * data[1][0] * data[2][1]
                    - data[0][2] * data[1][1] * data[2][0]
                    - data[0][0] * data[1][2] * data[2][1]
                    - data[0][1] * data[1][0] * data[2][2];
        inv[0][0] = (data[1][1] * data[2][2] - data[1][2] * data[2][1]) / det;
        inv[0][1] = -(data[1][0] * data[2][2] - data[1][2] * data[2][0]) / det;
        inv[0][2] = (data[1][0] * data[2][1] - data[1][1] * data[2][0]) / det;
        inv[1][0] = -(data[0][1] * data[2][2] - data[0][2] * data[2][1]) / det;
        inv[1][1] = (data[0][0] * data[2][2] - data[0][2] * data[2][0]) / det;
        inv[1][2] = -(data[0][0] * data[2][1] - data[0][1] * data[2][0]) / det;
        inv[2][0] = (data[0][1] * data[1][2] - data[0][2] * data[1][1]) / det;
        inv[2][1] = -(data[0][0] * data[1][2] - data[0][2] * data[1][0]) / det;
        inv[2][2] = (data[0][0] * data[1][1] - data[0][1] * data[1][0]) / det;
        return new Matrix(inv);
    }

    // return x = A^-1 b, assuming A is square and has full rank
    public Matrix solve(Matrix rhs) {
        if (M != N || rhs.M != N || rhs.N != 1)
            throw new RuntimeException("Illegal matrix dimensions.");

        // create copies of the data
        Matrix A = new Matrix(this);
        Matrix b = new Matrix(rhs);

        // Gaussian elimination with partial pivoting
        for (int i = 0; i < N; i++) {

            // find pivot row and swap
            int max = i;
            for (int j = i + 1; j < N; j++)
                if (Math.abs(A.data[j][i]) > Math.abs(A.data[max][i]))
                    max = j;
            A.swap(i, max);
            b.swap(i, max);

            // singular
            if (A.data[i][i] == 0.0) throw new RuntimeException("Matrix is singular.");

            // pivot within b
            for (int j = i + 1; j < N; j++)
                b.data[j][0] -= b.data[i][0] * A.data[j][i] / A.data[i][i];

            // pivot within A
            for (int j = i + 1; j < N; j++) {
                double m = A.data[j][i] / A.data[i][i];
                for (int k = i+1; k < N; k++) {
                    A.data[j][k] -= A.data[i][k] * m;
                }
                A.data[j][i] = 0.0;
            }
        }

        // back substitution
        Matrix x = new Matrix(N, 1);
        for (int j = N - 1; j >= 0; j--) {
            double t = 0.0;
            for (int k = j + 1; k < N; k++)
                t += A.data[j][k] * x.data[k][0];
            x.data[j][0] = (b.data[j][0] - t) / A.data[j][j];
        }
        return x;

    }

    // print matrix to standard output
    public void show(String TAG) {
        for (int i = 0; i < M; i++) {
            String row = "";
            for (int j = 0; j < N; j++)
                row += String.format("%9.4f, ", data[i][j]);
            Log.d(TAG, row);
        }
    }


}
