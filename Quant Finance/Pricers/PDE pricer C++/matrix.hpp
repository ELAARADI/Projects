#pragma once
#include <iostream>
#include <vector>


class matrix
{
    private:

    size_t m_nb_rows;
    size_t m_nb_cols;
    std::vector<double> m_data; // will contain matrix of size nb_rows*nb_columns

    public:

    //Constructors: 

    matrix();
    matrix(size_t nb_rows,size_t nb_cols); //constructor of matrix of 0
    matrix(size_t size); //constructor of square matrix

    //Constructor which takes a vector of double, a column vector
    matrix(std::vector<double> init); 

    //Getters: 

    size_t nb_rows() const;
    size_t nb_cols() const;

    // Operators:

    double& operator()(size_t i, size_t j); //returns a reference
    matrix& operator+=(matrix& rhs);
    matrix& operator*=(double rhs);

};

std::ostream& operator<<(std::ostream& out, matrix& m); //to print a matrix

matrix operator+(matrix lhs, matrix rhs); //to sum matrices
matrix operator*(int d,matrix m);
matrix operator*(matrix lhs,matrix rhs);


// Inverse a matrix
void inverse(matrix& A, matrix& inverse);
