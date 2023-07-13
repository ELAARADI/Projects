#include "blackscholes.hpp"



// Here we fill the matrices a b c and d in the case of a constant volatility 
void BS::fill_matrices(matrix& a,matrix& b, matrix& c, matrix& d, int n, int m,double vol){
    for (size_t i =0 ; i<n+1;i++){
        for (size_t j =0; j<m+1;j++){
            a(i,j) = 0;
            b(i,j) = -vol*vol/2;
            c(i,j) = vol*vol/2;
            d(i,j) = 0;
        }
    }
}

// Set the terminal condition (we use that x=log(s))
void BS::terminal_condition(std::vector<double>& v, int m, double S0,double K,double atm_volatility,double multiplier){
    std::vector<double> space_grid(m+1);
    for (size_t j = 0; j < m+1 ; j++){
        space_grid[j] = log(S0) - multiplier * atm_volatility + 2 * multiplier* atm_volatility * j/m; 
        if (exp(space_grid[j])> K){
            v[j]= exp(space_grid[j])-K;
        }
        else {
            v[j] = 0;
        }
    }
}

// Set the boundary condition
void BS::boundary_condition(std::vector<double>& v, int n,double T, double S0,double K,double atm_volatility,double multiplier,double r){
    std::vector<double> time_grid(n+1);
    double x= log(S0) + multiplier* atm_volatility;
    for (size_t i = 0; i < n+1 ; i++){
        time_grid[i] = i*T/n;
        v[i]= exp(x)-K*exp(-r*T);
    }
}

matrix BS::pricer_constante_volatility(double S0, double K, double r, double q, double T, double vol,int m, int n){
double multiplier = 5; 

matrix a(n+1,m+1);
matrix b(n+1,m+1);
matrix c(n+1,m+1);
matrix d(n+1,m+1);

// Constant volatility
fill_matrices(a,b,c,d,n,m,vol);
std::vector<double> vect_terminal_condition(m+1);
std::vector<double> vect_boundary_condition_1(n+1); // vector of 0
std::vector<double> vect_boundary_condition_2(n+1);
terminal_condition(vect_terminal_condition,m,S0,K,vol,multiplier);
boundary_condition(vect_boundary_condition_2,n,T,S0,K,vol,multiplier,r);
pricer p = pricer(m,n,1.0,vect_boundary_condition_1,vect_boundary_condition_2,vect_terminal_condition,a,b,c,d,0.3,S0);

std::string optionType = "call";
return p.price();
}

double BS::cdf_N(double x){
    return 0.5 * erfc(-x * M_SQRT1_2);
}

double BS::BS_price(std::string optionType, double S, double K, double r, double q, double T, double sigma){
    double d1 = (log(S/K) + (r-q+sigma*sigma*0.5)*(T))/(sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);

    if( optionType != "call" && optionType != "put")
        throw std::invalid_argument("The option type is not correct.");
   
    if(optionType == "call"){
        return S * exp(-q*T) * cdf_N(d1) - K * exp(-r* T) * cdf_N(d2);
    }
    else
        return K * exp(-r * (T)) * cdf_N(-d2) - S * exp(-q * (T)) * cdf_N(-d1);
}