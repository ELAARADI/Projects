#include "pricer.hpp"


// Vérifier les cas particuliers et les exceptions 
pricer::pricer(int _m,int _n,double _T,std::vector<double> boundary_conditions_1,std::vector<double> boundary_conditions_2,std::vector<double> _terminal_condition,matrix& _a,matrix& _b, matrix& _c,matrix& _d, double _atm_volatility, double _S0)
: m(_m)
, n(_n)
, T(_T)
, S0(_S0)
, boundary_conditions(boundary_conditions_1,boundary_conditions_2)
, terminal_condition(_terminal_condition)
, a(_a), b(_b), c(_c), d(_d)
, atm_volatility(_atm_volatility)
, theta(0.5)
, multiplier(5)
, time_grid(n+1)
, space_grid(m+1)
{
    if(m<= 0)
        throw std::invalid_argument("Space grid size is not correct.");
    if(n<=0)
        throw std::invalid_argument("Time grid size is not correct.");
    if(T<=0)
        throw std::invalid_argument("Time to maturity is not correct.");
    if(atm_volatility<0)
        throw std::invalid_argument("At the money volatility is not correct.");
    if(S0<=0)
        throw std::invalid_argument("Price is not correct.");  


    // We assume that a,b,c,d are of size (n+1)*(m+1), boundary_conditions is of size (n+1) (for both) and terminal condition is of size m+1

    std::cout << "Pricer construction \n";
    for (size_t i = 0; i<n+1;i++){
        time_grid[i] = i*T/n;
    }
    
    for (size_t j = 0; j < m+1 ; j++){
        space_grid[j] = log(S0) - multiplier * atm_volatility + 2 * multiplier* atm_volatility * j/m; // We center here on ln(S0)
    }
    init_P();
    init_Q();
    init_V();
    compute_U();
}
// We fill P: P[0] is the matrix P0
void pricer::init_P(){
    std::cout << "P initialization \n";
    double delta_t = T/n;
    double delta_x = 2 * multiplier* atm_volatility/m ; 
    for (size_t i = 0; i < n ;  i++){
        P.push_back(matrix(m-1,m-1));
        for (size_t k = 0; k<m-1;k++){
            if (k!=0){
                P[i](k,k-1)= -b(i,k+1)/(2*delta_x) + theta * c(i,k+1)/(delta_x*delta_x);
                }
            P[i](k,k) = -1/delta_t + a(i,1) - 2 * theta * c(i,k+1)/(delta_x*delta_x);
            if (k!=m-2){
                P[i](k,k+1)= b(i,k+1)/(2*delta_x) + theta * c(i,k+1)/(delta_x*delta_x);
                }
        }
    }
} 

void pricer::init_Q(){
    std::cout << "Q initialization \n";
    double delta_t = T/n;
    double delta_x = 2 * multiplier* atm_volatility/m ; 
    for (size_t i = 0; i<n; i++){
        Q.push_back(matrix(m-1,m-1));
        for (size_t k = 0; k<m-1;k++){
            if (k!=0){
                Q[i](k,k-1)= (1 - theta) * c(i,k+1)/(delta_x*delta_x);
                }
            Q[i](k,k) = 1/delta_t - 2 * (1- theta) * c(i,k+1)/(delta_x*delta_x);
            if (k!=m-2){
                Q[i](k,k+1)= (1- theta) * c(i,k+1)/(delta_x*delta_x);
                }
        }
    }
} 
void pricer::init_V(){
    std::cout << "V initialization \n";
    double delta_t = T/n;
    double delta_x = 2 * multiplier* atm_volatility/m ; 
    for (size_t i = 0; i< n; i++){
        V.push_back(matrix(m-1,1));
        V[i](0,0) =  d(i,1) + boundary_conditions.first[i]* (-b(i,1)/(2*delta_x)+ c(i,1)*theta/(delta_x*delta_x)) + boundary_conditions.first[i+1] * (1 - theta) * c(i,1)/(delta_x*delta_x);
        V[i](m-2,0) = d(i,m-1) + boundary_conditions.second[i]* (-b(i,m-1)/(2*delta_x)+ c(i,m-1)*theta/(delta_x*delta_x)) + boundary_conditions.second[i+1] * (1 - theta) * c(i,m-1)/(delta_x*delta_x);
        for(size_t k = 1; k<m-2;k++){
            V[i](k,0) = d(i,k+1);
        }
    }
} 

// We compute Un according to the terminal condition
void pricer::initialize_Un(){
    U[n]= matrix(m-1,1);
    for (size_t j = 0; j < m-1;j++){
        U[n](j,0)=terminal_condition[j+1];
    } 
}

// Backward computation of U
void pricer::compute_U(){
    std::cout << "U0 calculation \n";
    U = std::vector<matrix>(n+1); 
    initialize_Un();
    matrix mat_inverse= matrix(m-1,m-1);
    for (size_t i = 0;i < n;i++){
        inverse(P[n-1-i],mat_inverse); //we fill the inverse matrix 
        U[n-1-i] = (-1)* mat_inverse* ( ( Q[n-1-i] * U[n-i] ) + V[n-1-i] ) ;
    }
}

matrix pricer::price(){
    return U[0];
}
