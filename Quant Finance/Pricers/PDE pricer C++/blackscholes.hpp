#pragma once

#include "pricer.hpp"


class BS{

    public:

    //Pricer initialization:

    // We fill the matrices in the case where the volatility is constant
    void static fill_matrices(matrix& a,matrix& b, matrix& c, matrix& d, int n, int m,double vol);

    // We set the terminal condition
    void static terminal_condition(std::vector<double>& v, int m, double S0,double K,double atm_volatility,double multiplier);

    // We set the boundary condition
    void static boundary_condition(std::vector<double>& v, int n,double T, double S0,double K,double atm_volatility,double multiplier,double r);

    // This returns U0 when dealing with constant volatility 
    matrix static pricer_constante_volatility(double S0, double K, double r, double q, double T, double vol,int m, int n);

    double static cdf_N(double x);
    
    // This computes the Black-Scholes price 
    double static BS_price(std::string optionType, double S, double K, double r, double q, double timeToMaturity, double sigma);
};
