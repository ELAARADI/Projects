from math import sqrt
import numpy as np

class Matrix:
    def __init__(self,A:list) -> None:
        self.A = A
        self.n = len(A)
        
    def to_list(self):
        return self.A

    def Transpose(self):
        """ 
        Compute the transpose of a function : A.T -> AT 
        
        """
        if type(self.A[0]) == list: # Case where we're dealing with a list of lists
            AT = [[0 for j in range(self.n)] for i in range(len(self.A[0]))]
            for i in range(len(self.A[0])):
                for j in range(self.n):
                    AT[i][j] = self.A[j][i]
            return AT
        else: # Case where we're dealing with a list of numbers
            return self.A

    def isSymmetric(self):
        """
        Checks if the matrix is symmetric
        
        """
        tr = self.Transpose()
        return tr==self.A

    def is_invertible(self):
        """
        Computes the determinant of the matrix and returns a bool: 
            True if invertible
            False otherwise
        
        """
        # Compute the determinant of A
        det = Matrix.Determinant(self.A)

        # A is invertible if and only if its determinant is nonzero
        if det != 0:
            return True
        else:
            return False
        
    def matrix_product(A, B):
        """
        matrix product function: 
        -> considers the cases where A or B are vectors(list of numbers)
        
        Returns the result of the matricial product of the two input matrices
        
        """
        # Check if the number of columns of A matches the number of rows of B
        if type(A[0]) == list:  #checks the case where A is not a vector
            if len(A[0]) != len(B):
                raise ValueError("Number of columns of A must match number of rows of B")
        
            # Create the result matrix
            if type(B[0]) == list: #checks the case where B is not a vector
                product = [[0 for j in range(len(B[0]))] for i in range(len(A))]
                # Compute the matrix product
                for i in range(len(A)):
                    for j in range(len(B[0])):
                        for k in range(len(B)):
                            product[i][j] += A[i][k] * B[k][j]
                return product
            
            else:
                # Compute the product of A and B(a vector)
                product = []
                for row in A:
                    row_res = 0
                    for j in range(len(B)):
                        row_res += row[j] * B[j]
                    product.append(row_res)
                return product
            
        else:
            if len(A) != len(B):
                raise ValueError("Number of columns of A must match number of rows of B")
            
            # Create the result matrix
            if type(B[0]) == list: #checks the case where B is not a vector
                product = [0 for j in range(len(B[0]))]
                # Compute the matrix product
                for i in range(len(B[0])):
                    for k in range(len(A)):
                        product[i] += A[k] * B[k][i]
                return product
            
            else:
                # Compute the product of A(a vector) and B(a vector)
                product = 0
                for i in range(len(A)):
                    product += A[i] * B[i]
                return product
        
        

    def identity(size:int):
        """
        Identity matrix constructor
        
        """
        I = []
        for i in range(size):
                row = []
                for j in range(size):
                    if i == j:
                        row.append(1)
                    else:
                        row.append(0)
                I.append(row)
        return I

    def Determinant(matrix:list)->float:
        """
        Returns the determinant of an input square matrix.
        -> recursive algorithm to compute the determinant of the matrix.
        
        """
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        else:
            det = 0
            for j in range(n-1):
                minor = [row[:j] + row[j+1:] for row in matrix[1:]]
                det += ((-1) ** j) * matrix[0][j] * Matrix.Determinant(minor)
            return det
        
    def cofactor(matrix:list)->list:
        """ 
        This function computes the cofactor matrix of the input matrix. 
        It is used later to determine the eigen values of the input matrix.
        
        """
        n = len(matrix)
        cofactor_matrix = [[0] * n for i in range(n)]
        for i in range(n):
            for j in range(n):
                minor = [row[:j] + row[j+1:] for row in matrix[:i] + matrix[i+1:]]
                cofactor_matrix[i][j] = ((-1) ** (i+j)) * Matrix.Determinant(minor)
        return cofactor_matrix

    def concat(matrix, identity, axis=0):
        """
        The function returns a new matrix that is the result of concatenating the two input matrices along the specified axis.
            If axis=0, the function concatenates the two matrices vertically.
            If axis=1, the function concatenates the two matrices horizontally.
        
        """
        if axis == 0:
            return matrix + identity
        elif axis == 1:
            return [row + identity[i] for i, row in enumerate(matrix)]
        else:
            raise ValueError("axis must be either 0 or 1")

    def inverse_matrix(matrix):
        """
        Equivalent to the np.linalg.inv function to up to 16 decimals
        
        
        """
        
        # Check if the matrix is square
        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix must be square")

        # Create an augmented matrix with the identity matrix on the right
        n = len(matrix)
        aug_matrix = Matrix.concat(matrix, Matrix.identity(n), axis=1)

        # Perform row operations to get the upper triangular matrix
        for i in range(n):
            if aug_matrix[i][i] == 0:
                raise ValueError("Matrix is not invertible")
            for j in range(i+1, n):
                ratio = aug_matrix[j][i] / aug_matrix[i][i]
                for k in range(len(aug_matrix[j])):
                    aug_matrix[j][k] -= ratio * aug_matrix[i][k]

        # Perform row operations to get the identity matrix on the left
        for i in range(n-1, -1, -1):
            for j in range(i-1, -1, -1):
                ratio = aug_matrix[j][i] / aug_matrix[i][i]
                for k in range(len(aug_matrix[j])):
                    aug_matrix[j][k] -= ratio * aug_matrix[i][k]

        # Scale each row to make the diagonal entries equal to 1
        for i in range(n):
            if aug_matrix[i][i] == 0:
                raise ValueError("Matrix is not invertible")
            aug_matrix[i] = list(np.array(aug_matrix)[i] / aug_matrix[i][i]) 
            #doesn't work in list format need to use the numpy array for some reason


        # Extract the right half of the augmented matrix as the inverse
        inverse = [aug_matrix[i][n:] for i in range(len(aug_matrix))]
        return inverse


class Mx_decomp:
    def __init__(self, A:Matrix):
        self.A = A
        self.n = len(A.to_list())
    
    def Decompose(self):
        """
        Takes as input a covariance matrix in list format.
        
        - We check for the first condition (symmetry of A)  -> fundamental
        - We then check if A is invertible
            - if so: Perform a Cholesky decomposition of A, which is a symmetric 
            and positive definite matrix. 
            - if not: We go through the diagonalization process.
            
        The diagonalization process is in two versions below:
            - one using numpy (by necessity)
            - the second (function diagonalize()) where we coded everything from scratch 
            but sadly doesn't work due to the eigenvalues function
        
        The function returns :
        - the lower variant triangular matrix L when going through the Cholesky transformation
        - the product of the orthogonal matrix and the square root of the diagonal matrix otherwise
        
        """
        L = [[0.0] * self.n for i in range(self.n)]  #zero matrix to receive L
       
        # Check the conditions to apply the decomposition
        if (not self.A.isSymmetric()):
            print("Matrix is not symmetric, decomposition is impossible.")
            return L
        
        if self.A.is_invertible():
            # Perform the Cholesky decomposition
            for i in range(self.n):
                for k in range(i+1):
                    tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
                    
                    if (i == k): # Diagonal elements
                        L[i][k] = sqrt(self.A.to_list()[i][i] - tmp_sum)
                    else:
                        L[i][k] = (1.0 / L[k][k] * (self.A.to_list()[i][k] - tmp_sum))
            return L

        
        elif all(np.linalg.eigvals(self.A)) > 0:
            # Resort to Numpy's diagonalization functions (our implementation below isn't very accurate)
            # Compute the eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eig(self.A.to_list())

            # Check if the matrix is diagonalizable
            if np.linalg.matrix_rank(eigvecs) == np.array(self.A.to_list()).shape[0]:
                print("The matrix is diagonalizable.")

                # Compute the diagonal matrix and the matrix of eigenvectors
                D = np.diag(eigvals)
                P = eigvecs
            
            # Check if the product matches with the matrix in input
            return np.dot(P,np.sqrt(D))
        
        elif (not self.A.is_invertible()) and (not self.IsPositiveDefinite()):
            print("Matrix is neither invertible nor positive definite, decomposition impossible.")
            return L

    def eigenvalues(self)->list:
        """ Compute the eigen values of a matrix and store it into a list"""
        eigvals = []
        for i in range(self.n):
            cofactor_matrix = Matrix.cofactor([[self.A.to_list()[j][k] for k in range(self.n) if k != i] for j in range(self.n) if j != i])
            eigvals.append(self.A.to_list()[i][i] * Matrix.Determinant(cofactor_matrix))
        return eigvals
    
    def diagonalize(self):
        """
        Executes the diagonalization process: takes as input the matrix to diagonalize.
        Returns the product of the orthogonal matrix and the square root of the diagonal matrix 
        
        """
        eigvals = self.eigenvalues()

        # Check if the matrix is diagonalizable
        eigvecs = []
        for eigval in eigvals:
            eigenspace = [list(map(lambda x: x-eigval, row)) for row in self.A.to_list()]
            ker = []
            for i in range(len(eigenspace)):
                row = [1 if i == j else 0 for j in range(len(eigenspace))]
                row.append(0)
                ker.append(row)
            ker.append([0 for i in range(len(eigenspace))] + [1])
            eigenvectors = []
            for i in range(len(eigenspace)):
                submatrix = [row[:i] + row[i+1:] for row in eigenspace[:i] + eigenspace[i+1:]]
                subdet = Matrix.Determinant(submatrix)
                for j in range(len(eigenspace)):
                    ker[j][i] = submatrix[j-1][j-1]/subdet
                eigenvectors.append(ker[i][:len(eigenspace)])
            eigvecs.append(eigenvectors)

        P = [] #the orthogonal matrix
        for i in range(len(eigvecs[0])):
            row = []
            for j in range(len(eigvecs[0])):
                row.append(eigvecs[j][i])
            P.append(row)
        P = P[-1]
            
        D = [] #the diagonal matrix
        for i in range(len(eigvals)):
            row = []
            for j in range(len(eigvals)):
                if i == j:
                    row.append(eigvals[j])
                else:
                    row.append(0)
            D.append(row)
        
        result = [] #the matricial product of P with the sqroot of D
        for i in range(len(D)):
            row = []
            for j in range(len(D)):
                row.append(P[i][j]*sqrt(D[j][j]))
            result.append(row)
        
        return result

    def IsPositiveDefinite(self)->list:
        return all(self.eigenvalues()) > 0
    
    
    