#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <vector>

using namespace std;

class Matrix {
 public:

  // Initialization
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        data_[i * cols_ + j] = 0.0;}}
  }
  Matrix(int rows, int cols, double value) :   rows_(rows),   cols_(cols),   data_(rows * cols) {
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        data_[i * cols_ + j] = value;}}
  }

  double &operator()(int row, int col) {
    return data_[row * cols_ + col];
  }

  double operator()(int row, int col) const {
    return data_[row * cols_ + col];
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }

  // To multiply matrices
  Matrix operator*(const Matrix& other) const 
  {
    Matrix result(rows_, other.cols());
    for (int i = 0; i < result.rows(); i++) {
      for (int j = 0; j < result.cols(); j++) {
        double sum = 0.0;
        for (int k = 0; k < cols_; k++) {
          sum += (*this)(i, k) * other(k, j);
        }
        result(i, j) = sum;
      }
    }
    return result;
  }

  // To multiply matrix by integer
  Matrix operator*(double x) const 
  {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        result(i, j) = x * (*this)(i, j);
      }
    }
    return result;
  }

  // To add matrices
  Matrix operator+(const Matrix& other) const 
  {
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        result(i, j) = (*this)(i, j) + other(i, j);
      }
    }
    return result;
  }

  // To initialize matrix to 0
  void init_0()
  {
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        (*this)(i, j) = 0.0;
      }
    }
  }

  // To extact a column vector
  Matrix extract_col(int my_col) const 
  {
        Matrix result(rows_, 1);
                for (int i = 0; i < rows_; i++) {
            result(i, 0) = (*this)(i, my_col);
        }
        return result;
    }
  
  // To extact a row from a matrix
  Matrix extract_row(int my_row) const 
  {
        Matrix result(cols_, 1, 0.0);
        for (int i = 0; i < cols_; i++) {
            result(i, 0) = double((*this)(my_row, i));
        }
        return result;
    }

  // To add a column vector
  void add_vector(Matrix& other, int my_row)
  {
    for (int i = 0; i < cols_; i++) {
      (*this)(my_row, i) += other(i, 0);
    }
  }

  // To add a row vector
  void add_row_vector(Matrix& other, int my_col)
  {
    for (int i = 0; i < rows_; i++) {
      (*this)(i, my_col) += other(i, 0);
    }
  }
  
  Matrix inverse() const {
    if (rows_ != cols_) {
        // Throw an error or return a special value
        cout << "size error";
    }

    Matrix I(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        I(i, i) = 1.0;
    }

    Matrix A_inv = I;
    Matrix A = *this;

    // Gaussian-Jordan elimination
    for (int i = 0; i < rows_; i++) {
        // Find pivot
        int pivot = i;
        for (int j = i + 1; j < rows_; j++) {
            if (fabs(A(j, i)) > fabs(A(pivot, i))) {
                pivot = j;
            }
        }

        // Swap rows
        if (pivot != i) {
            for (int j = 0; j < cols_; j++) {
                double temp = A(i, j);
                A(i, j) = A(pivot, j);
                A(pivot, j) = temp;

                temp = A_inv(i, j);
                A_inv(i, j) = A_inv(pivot, j);
                A_inv(pivot, j) = temp;
            }
        }

        // Normalize pivot row
        double pivot_val = A(i, i);
        for (int j = 0; j < cols_; j++) {
            A(i, j) /= pivot_val;
            A_inv(i, j) /= pivot_val;
        }

        // Divide pivot row by pivot element
        double pivot_inv = 1.0 / A(i, i);
        A(i, i) = 1.0;
        for (int j = i + 1; j < cols_; j++) {
            A(i, j) *= pivot_inv;
        }
        for (int j = 0; j < cols_; j++) { 
            A_inv(i, j) *= pivot_inv;
        }

        // Eliminate pivot column
        for (int j = 0; j < rows_; j++) {
            if (i != j) {
                double factor = A(j, i);
                for (int k = 0; k < cols_; k++) {
                    A(j, k) -= factor * A(i, k);
                    A_inv(j, k) -= factor * A_inv(i, k);
                }
            }
        }
    }
    return A_inv;
  }

  // To display matrix
  void display()
  {
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        cout << (*this)(i, j) << " ";
      }
      cout << endl;
    }
  }

 private:
  int rows_;
  int cols_;
  vector<double> data_;
};



// Functions to compute B&S price
double normalCDF(double x)
{
    return erfc(-x/sqrt(2))/2;
}

// Usual B&S function to price a call
double black_scholes_call(double S, double K, double r, double sigma, double T, string type) {
  double d1 = (log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T));
  double d2 = d1 - sigma * sqrt(T);
  if (type == "Call")
    return (S * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2));
  else
    return (-S * normalCDF(-d1) + K * exp(-r * T) * normalCDF(-d2));
}


// Function to calculate the value of vanilla option using PDE approach
double PDE_Vanilla_Option(double S, double K, double r, double T, double sigma, int N, int M, string option_type, 
                          double low_b[], double up_b[], int lambda, Matrix A, Matrix B, Matrix C, Matrix D)
{
	// create time grid
	double dt = T/(N-1);
  double *time_grid = new double[N];
  time_grid[0] = 0.00;
	for (int i = 1; i < N; i++){
		time_grid[i] = time_grid[i-1] + dt;}


  // Create Space grid : log(S/F)
  double dS = S*2*lambda*sigma/(M-1);
  double *space_grid = new double[M];
  double *dx = new double[M];
  double S_min = S*(1-lambda*sigma);
  double St = S_min + dS;
  space_grid[0] = log(S_min/S);
	for (int i = 1; i < M; i++){
    St = S_min + dS * i;
		space_grid[i] = log(St/S);
    dx[i-1] = space_grid[i] - space_grid[i-1];
  }
  dx[M-1] = log((S_min + dS * M)/S)-space_grid[M-1];

  

  Matrix U(N, M, 0.0);

	// Set terminal condition
	for (int j = 0; j < M-1; j++){
        if (option_type=="Call"){
            U(N-1, j) = max(0.0, ((S_min+dS*(j)) - K));
        } else {
            U(N-1, j) = max(0.0, (K - (S_min+dS*(j))));
        }
    }
  

    //Set Dirichlet boundary conditions
    for (int i = 0; i < N; i++){
        U(i, 0) = low_b[i];  //correspond to low boundary
        U(i, M-1) = up_b[i];  //correspond to up boundary
    }

    // Possible to change the value of theta
    double theta(0.5);

	// Calculate option values
  Matrix P(M-1, M-1, 0.0);
  Matrix Q(M-1, M-1, 0.0);
  Matrix V(M-1, 1, 0.0);
  Matrix temp(M-1, M-1, 0.0);
  Matrix temp2(M-1, 1, 0.0);
  Matrix temp_U(M-1, 1, 0.0);


	for (int i = N-2; i >= 0; i--)
	{
  // Reset the matrix at each time step
    P.init_0();
    Q.init_0();
    V.init_0();

		for (int j = 0; j < M-1; j++)
		{ 
			P(j, j) = A(i, j) -1/dt - 2*theta*C(i, j)/(dx[j]*dx[j+1]);
      Q(j, j) = 1/dt - 2*(1-theta)*C(i, j)/(dx[j]*dx[j+1]);
      V(j, 0) = D(i, j);

      if (j > 0){
          P(j, j-1) = -B(i, j)/(dx[j]+dx[j+1]) + (2*theta*C(i, j))/(dx[j]*dx[j+1]);
          Q(j, j-1) = (1-theta)*C(i, j)/(dx[j]*dx[j+1]);
      } 
      else {
          V(j, 0) = D(i, j) + ( -B(i, j)/(dx[j]+dx[j+1]) + (2*theta*C(i, j))/(dx[j]*dx[j+1]) ) * U(i, 0) + ( (1-theta)*C(i, j)/(dx[j]*dx[j+1]) ) * U(i+1, 0);
      }

      if (j < M-2){
          P(j, j+1) = B(i, j)/(dx[j]+dx[j+1]) + (2*theta*C(i, j))/(dx[j]*dx[j+1]);
          Q(j, j+1) = (1-theta)*C(i, j)/(dx[j]*dx[j+1]);
      } else {
          V(j, 0) = D(i, j) + ( -B(i, j)/(dx[j]+dx[j+1]) + (2*theta*C(i, j))/(dx[j]*dx[j+1]) ) * U(i, M) + ( (1-theta)*C(i, j)/(dx[j]*dx[j+1]) ) * U(i+1, M);
      }

		}
    
    temp = P.inverse()* -1;
    temp2 = Q*U.extract_row(i+1) + V;
    temp_U = temp*temp2;

    for (int a = 1; a < M-1; a++) {
       if (temp_U(a, 0) < 0) temp_U(a, 0) = 0.0;
    }

    for (int a = 1; a < M-1; a++) { 
      U(i, a) = temp_U(a, 0);
    }
	}

	// Calculate option value
  U.display();
  return U(0, int(M/2));
}

int main()
{
  // Declare variables
	double S, K, r, T, sigma;
	int N, M;
  string option_type;
	
	// // Input option parameters
	// cout << "Enter stock price S: ";
	// cin >> S;
	// cout << "Enter strike price K: ";
	// cin >> K;
	// cout << "Enter risk free rate r: ";
	// cin >> r;
	// cout << "Enter time to maturity T: ";
	// cin >> T;
	// cout << "Enter volatility sigma: ";
	// cin >> sigma;
	// cout << "Enter number of steps M: ";
	// cin >> M;
	// cout << "Enter number of time steps N: ";
	// cin >> N;
  // cout << "Enter type of option (Call or Put): ";
	// cin >> option_type;

    S = 100;
    K = 100;
    r = 0.05;
    T = 1.0;
    sigma = 0.16;
    M = 11;
    N = 100;
    option_type = "Call";

    // Create the matrices
    Matrix A(N,M,0.0);
    Matrix B(N,M,0.5*sigma*sigma);
    Matrix C(N,M,-0.5*sigma*sigma);
    Matrix D(N,M,0.0);

    //Define Lambda for the space grid boundary
    int lambda = 5;
    double S_min = S*(1-lambda*sigma);
    double S_max = S*(1+lambda*sigma);
    
    //Create the Dirichlet boundaries
    double* low_b = new double[N];
    double* up_b = new double[N];
    double dS = S*2*lambda*sigma/(M-1);


    for(int i=0; i<N; i++)
    {
        if (option_type == "Call"){
            low_b[i] = 0.00;
            up_b[i] = max(0.0, ((S_max) - K*exp(-r*(T-i*T/N))));
        } else{
            low_b[i] = max(0.0, (K*exp(-r*(T-i*T/N)) - S_min));
            up_b[i] = 0.00;
        }
    }

    double PDE_price = PDE_Vanilla_Option(S,K,r,T,sigma,N,M,option_type, low_b, up_b,lambda,A,B,C,D);
    double priceBS = black_scholes_call(S,K,r,sigma,T,option_type);
    cout << "BS Price: " << priceBS << endl;
    cout << "PDE Price: " << PDE_price << endl;
    return 0;
}