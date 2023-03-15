#include "pricer.hpp"
#include "blackscholes.hpp"


int main(int argc, char *argv[]) {
    double T = 1.0;
    double vol = 0.2;
    double K = 100.0;
    double S0 = 100.0;
    double r = 0.05;
    double q = 0.0; 

    int m = 10;
    int n = 100;

    try{
        matrix U0= BS::pricer_constante_volatility(S0,K,r,q,T,vol,m,n);
        std::cout << U0 << "\n";

        double v = BS::BS_price("call",S0,K,r,q,T,vol);
        std::cout << "BS price is " << v <<" \n";
    }
    catch(std::invalid_argument& e){
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}


