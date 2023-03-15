#pragma once

#include "matrix.hpp"
#include <utility>
#include <math.h>


class pricer
{
    private: 

    int m;
    int n;
    double T;
    double theta;

    double atm_volatility;

    double multiplier;

    double S0;

    std::vector<double> time_grid;
    std::vector<double> space_grid;

    std::pair<std::vector<double>, std::vector<double>> boundary_conditions; //of size n+1

    std::vector<double> terminal_condition; // of size m+1

    matrix a,b,c,d; // We assume that the matrix is defined on [0,T] x [xm, xM] i.e. matrix is of size (n+1)*(m+1)

    std::vector<matrix> P; //vector which contains matrices Pi
    
    std::vector<matrix> Q; 
    
    std::vector<matrix> V;

    std::vector<matrix> U;

    void init_P(); //fills the vector P (P[0] is the matrix P0)
 
    void init_Q();

    void init_V();

    void compute_U(); //calculation of U (backward construction)

    void initialize_Un();

    public:

    pricer(int _m,int _n,double _T,std::vector<double> boundary_conditions_1,std::vector<double> boundary_conditions_2,std::vector<double> _terminal_condition,matrix& a,matrix& b, matrix& c,matrix& d, double _atm_volatility, double _S0);

    matrix price();

};
