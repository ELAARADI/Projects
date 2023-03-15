#include "matrix.hpp"


//Constructors: 

matrix::matrix()
: m_nb_rows(0),
  m_nb_cols(0),
  m_data()
{}

matrix::matrix(size_t nb_rows, size_t nb_cols)
: m_nb_rows(nb_rows),
  m_nb_cols(nb_cols),
  m_data(nb_rows * nb_cols)
{}
 
matrix::matrix(size_t size)
: matrix(size,size)
{}

matrix::matrix(std::vector<double> init): 
m_nb_rows(init.size()),
m_nb_cols(1),
m_data(init.size())
{
  size_t size = init.size();
  for (size_t i = 0; i< size; i++){
    m_data[i]= init[i];
  }
}
 
//Getters: 

size_t matrix::nb_rows() const
{
  return m_nb_rows;
}

size_t matrix::nb_cols() const
{
  return m_nb_cols;
}    

//Operators: 

double& matrix::operator()(size_t i, size_t j)
{
  return m_data[i * m_nb_cols + j];
}

matrix& matrix::operator+=(matrix& rhs) //We assume same size
{

  for(std::size_t i = 0; i < nb_rows(); ++i)
  {
    for(std::size_t j = 0; j < nb_cols(); ++j)
    {
        m_data[i * m_nb_cols + j]+=rhs.m_data[i*m_nb_cols+j];
    }
  }
  return *this;
}

matrix& matrix::operator*=(double rhs)
{
  for(std::size_t i = 0; i < nb_rows(); ++i)
  {
    for(std::size_t j = 0; j < nb_cols(); ++j)
    {
        m_data[i * m_nb_cols + j]*=rhs;
    }
  }
  return *this;
}
   
std::ostream& operator<<(std::ostream& out, matrix& m)
{
  for(std::size_t i = 0; i < m.nb_rows(); i++)
  {
    for(std::size_t j = 0; j < m.nb_cols(); j++)
    {
      out << m(i, j) << ", ";
    }
    out << std::endl;
  }
  return out;
}

matrix operator+(matrix lhs,matrix rhs)
{
  matrix tmp(lhs);
  tmp += rhs;
  return tmp;
}

matrix operator*(matrix lhs,matrix rhs) //multiply A by B
{
  matrix tmp(lhs.nb_rows(),rhs.nb_cols()); 
  for(std::size_t i = 0; i < lhs.nb_rows(); i++)
  {
    for(std::size_t j = 0; j < rhs.nb_cols(); j++)
    {
      for(std::size_t k = 0; k< lhs.nb_cols();k++)
      {
        tmp(i,j)=tmp(i,j)+(lhs(i,k)*rhs(k,j));
      }
    }
  }
  return tmp;
}

matrix operator*(int d,matrix m)
{
  matrix tmp(m); 
  tmp *=d;
  return tmp;
}

//We apply the Gauss elimination algorithm to find the inverse of a matrix m :

void inverse(matrix& m, matrix& inverse){ 
  size_t i,j,k;
  double p; //r
  size_t n = m.nb_rows();
  //Augmented matrix of m : 
  matrix A(n,2*n); 
  // Copy of the matrix m on the right of matrix A : 
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      A(i,j)=m(i,j);
    }
  }
  // We complete by the identity matrix on the left of matrix A :
  for(i=0;i<n;i++){
    for(j=n;j<2*n;j++){
      if(i==j-n) A(i,j)=1;
      else A(i,j)=0;
    }
  }
  // We apply here the algorithm of Gauss elimination :
  for(i=0;i<n;i++){
    p=A(i,i); //pivot
    for(j=i;j<2*n;j++){
      A(i,j)=A(i,j)/p;
    }
    for(j=0;j<n;j++){
      if(i!=j){
        p=A(j,i);
        for(k=0;k<2*n;k++)
        A(j,k)=A(j,k)-p*A(i,k);
      }
    } 
  }
  // End of the Gauss elimination algorithm: we copy the right matrix in the inverse matrix :
  for(i=0;i<n;i++){
    for(j=n;j<2*n;j++){
      inverse(i,j-n)=A(i,j);
    }  
  }
}
