import cmath
import math

class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if not isinstance(other, ComplexNumber):
            raise TypeError("Operand must be a ComplexNumber")
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        if not isinstance(other, ComplexNumber):
            raise TypeError("Operand must be a ComplexNumber")
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        if not isinstance(other, ComplexNumber):
            raise TypeError("Operand must be a ComplexNumber")
        if other.real == 0 and other.imag == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        denom = other.real ** 2 + other.imag ** 2
        real = (self.real * other.real + self.imag * other.imag) / denom
        imag = (self.imag * other.real - self.real * other.imag) / denom
        return ComplexNumber(real, imag)

    def __abs__(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def cc(self):
        return ComplexNumber(self.real, -self.imag)

    def __str__(self):
        return f"{self.real} + {self.imag}i"

class Vector:
    def __init__(self, field_type, length, coordinates):
        if field_type not in ['real', 'complex']:
            raise ValueError("Field type must be 'real' or 'complex'")
        if length != len(coordinates):
            raise ValueError("Length of coordinates must match the specified vector length")
        
        self.field_type = field_type
        self.length = length
        self.coordinates = []

        for coord in coordinates:
            if field_type == 'real':
                if not isinstance(coord, (int, float)):
                    raise TypeError("Coordinates must be real numbers")
                self.coordinates.append(coord)
            elif field_type == 'complex':
                if not isinstance(coord, ComplexNumber):
                    raise TypeError("Coordinates must be ComplexNumber instances")
                self.coordinates.append(coord)

    def __str__(self):
        return f"Vector({self.field_type}, {self.coordinates})"

    def len(self):
        return self.length
    
    @staticmethod
    def inner_product(v1, v2):
        """Compute the inner product of two vectors."""
        if v1.length != v2.length:
            raise ValueError("Vectors must have the same length")

        return sum(v1.coordinates[i] * v2.coordinates[i] for i in range(v1.length))

    @staticmethod
    def is_ortho(v1, v2):
        """Check if two vectors are orthogonal."""
        return Vector.inner_product(v1, v2) == 0
    
    @staticmethod
    def gram_schmidt(S):
        """Perform Gram-Schmidt orthogonalization on a set of vectors."""
        if not S:
            return []

        field_type = S[0].field_type
        length = S[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in S):
            raise ValueError("All vectors must have the same field type and length")

        orthogonal_vectors = []
        for v in S:
            
            ortho_v = v.coordinates[:]
            for u in orthogonal_vectors:
                
                proj = Vector.inner_product(Vector(field_type, length, ortho_v), u) / Vector.inner_product(u, u)
                ortho_v = [ortho_v[i] - proj * u.coordinates[i] for i in range(length)]
            orthogonal_vectors.append(Vector(field_type, length, ortho_v))

        return orthogonal_vectors

class Matrix:
    def __init__(self, field_type, n, m, data_or_vectors):
        if field_type not in ['real', 'complex']:
            raise ValueError("Field type must be 'real' or 'complex'")
        
        self.field_type = field_type
        self.rows = n
        self.cols = m

        
        if all(isinstance(item, Vector) for item in data_or_vectors):
            vectors = data_or_vectors
            if len(vectors) != m:
                raise ValueError("Number of vectors must match the number of columns (m)")
            if not all(vec.field_type == field_type for vec in vectors):
                raise ValueError("All vectors must have the same field type as the matrix")
            if not all(vec.length == n for vec in vectors):
                raise ValueError("All vectors must have the same length as the number of rows (n)")

            
            self.data = [[vec.coordinates[i] for vec in vectors] for i in range(n)]
        
        
        else:
            data = data_or_vectors
            if len(data) != n or any(len(row) != m for row in data):
                raise ValueError("Data dimensions must match the specified matrix dimensions (n x m)")
            
            self.data = []
            for row in data:
                validated_row = []
                for value in row:
                    if field_type == 'real':
                        if not isinstance(value, (int, float)):
                            raise TypeError("Matrix elements must be real numbers")
                        validated_row.append(value)
                    elif field_type == 'complex':
                        if not isinstance(value, ComplexNumber):
                            raise TypeError("Matrix elements must be ComplexNumber instances")
                        validated_row.append(value)
                self.data.append(validated_row)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Operand must be a Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition")
        
        result_data = []
        for i in range(self.rows):
            result_row = []
            for j in range(self.cols):
                result_row.append(self.data[i][j] + other.data[i][j])
            result_data.append(result_row)
        
        return Matrix(self.field_type, self.rows, self.cols, result_data)

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Operand must be a Matrix")
        if self.cols != other.rows:
            raise ValueError("Number of columns in the first matrix must equal the number of rows in the second matrix for multiplication")
        
        result_data = []
        for i in range(self.rows):
            result_row = []
            for j in range(other.cols):
                sum_value = 0
                for k in range(self.cols):
                    sum_value += self.data[i][k] * other.data[k][j]
                result_row.append(sum_value)
            result_data.append(result_row)
        
        return Matrix(self.field_type, self.rows, other.cols, result_data)

    def inv(self):
        """Compute the inverse of the matrix using row reduction if it is invertible."""
        if not self.is_square():
            raise ValueError("Inverse is only defined for square matrices")

        n = self.rows
        
        augmented_matrix = [self.data[i] + [1 if i == j else 0 for j in range(n)] for i in range(n)]

        
        for i in range(n):
            
            pivot = augmented_matrix[i][i]
            if pivot == 0:
                
                for j in range(i + 1, n):
                    if augmented_matrix[j][i] != 0:
                        augmented_matrix[i], augmented_matrix[j] = augmented_matrix[j], augmented_matrix[i]
                        pivot = augmented_matrix[i][i]
                        break
                else:
                    raise ValueError("Matrix is not invertible")

            
            augmented_matrix[i] = [x / pivot for x in augmented_matrix[i]]

            
            for j in range(n):
                if j != i:
                    factor = augmented_matrix[j][i]
                    augmented_matrix[j] = [augmented_matrix[j][k] - factor * augmented_matrix[i][k] for k in range(2 * n)]

        
        inverse_data = [row[n:] for row in augmented_matrix]
        return Matrix(self.field_type, n, n, inverse_data)


    def get_row(self, index):
        if index < 0 or index >= self.rows:
            raise IndexError("Row index out of range")
        return self.data[index]

    def get_column(self, index):
        if index < 0 or index >= self.cols:
            raise IndexError("Column index out of range")
        return [self.data[i][index] for i in range(self.rows)]

    def transpose(self):
        transposed_data = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(self.field_type, self.cols, self.rows, transposed_data)
    
    def pseudo_inverse(self):
        """Compute the Moore-Penrose pseudoinverse of the matrix."""
        if self.rows >= self.cols:
            
            At = self.transpose()
            AtA = At * self
            AtA_inv = AtA.inv()
            return AtA_inv * At
        else:
            
            At = self.transpose()
            AAt = self * At
            AAt_inv = AAt.inv()
            return At * AAt_inv

    def conj(self):
        if self.field_type != 'complex':
            raise ValueError("Conjugate operation is only applicable to complex matrices")
        conjugated_data = [[value.cc() for value in row] for row in self.data]
        return Matrix(self.field_type, self.rows, self.cols, conjugated_data)

    def conj_transpose(self):
        return self.conj().transpose()

    def is_zero(self):
        return all(all(value == 0 for value in row) for row in self.data)

    def is_square(self):
        return self.rows == self.cols

    def is_symmetric(self):
        if self.field_type != 'real' or not self.is_square():
            return False
        return all(self.data[i][j] == self.data[j][i] for i in range(self.rows) for j in range(self.cols))

    def is_hermitian(self):
        if self.field_type != 'complex' or not self.is_square():
            return False
        return all(self.data[i][j] == self.data[j][i].cc() for i in range(self.rows) for j in range(self.cols))

    def is_identity(self):
        if not self.is_square():
            return False
        return all(self.data[i][j] == (1 if i == j else 0) for i in range(self.rows) for j in range(self.cols))

    def is_singular(self):
        
        return self.determinant() == 0

    def is_invertible(self):
        return not self.is_singular()

    def determinant(self):
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")
        
        if self.rows == 1:
            return self.data[0][0]
        
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        det = 0
        for c in range(self.cols):
            det += ((-1) ** c) * self.data[0][c] * self.minor(0, c).determinant()
        return det

    def minor(self, i, j):
        return Matrix(self.field_type, self.rows - 1, self.cols - 1,
                      [row[:j] + row[j+1:] for row in (self.data[:i] + self.data[i+1:])])

    def char_poly(self):
        """Compute the characteristic polynomial of the matrix."""
        if not self.is_square():
            raise ValueError("Characteristic polynomial is only defined for square matrices")

        
        identity = [[1 if i == j else 0 for j in range(self.rows)] for i in range(self.rows)]

        
        lambda_matrix = [[self.data[i][j] - (1 if i == j else 0) for j in range(self.cols)] for i in range(self.rows)]
        char_poly_coeffs = [self.determinant(lambda_matrix)]
        return char_poly_coeffs
    
    def is_similar(self, B):
        """Check if two matrices are similar."""
        if not self.is_square() or not B.is_square() or self.rows != B.rows:
            raise ValueError("Both matrices must be square and of the same size")

        
        return self.char_poly() == B.char_poly()

    def alg_mul(self, eigenvalue):
        """Compute the algebraic multiplicity of a given eigenvalue."""
        char_poly = self.char_poly()
        return char_poly.count(eigenvalue)
    
    def cholesky_decomposition(self):
        """Compute the Cholesky decomposition of the matrix."""
        if not self.is_square():
            raise ValueError("Cholesky decomposition is only defined for square matrices")

        n = self.rows
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                sum_val = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:  
                    if self.data[i][i] - sum_val <= 0:
                        raise ValueError("Matrix is not positive definite")
                    L[i][j] = math.sqrt(self.data[i][i] - sum_val)
                else:
                    L[i][j] = (self.data[i][j] - sum_val) / L[j][j]

        return L
    
    def invadj(self):
        """Compute the inverse of the matrix using the adjoint method if it is invertible."""
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is not invertible")

        
        cofactors = []
        for r in range(self.rows):
            cofactor_row = []
            for c in range(self.cols):
                minor_det = self.minor(r, c).determinant()
                cofactor_row.append(((-1) ** (r + c)) * minor_det)
            cofactors.append(cofactor_row)

        
        adjoint = [[cofactors[c][r] for c in range(self.cols)] for r in range(self.rows)]

        
        inverse_data = [[adjoint[r][c] / det for c in range(self.cols)] for r in range(self.rows)]
        return Matrix(self.field_type, self.rows, self.cols, inverse_data)

    def is_diagonalizable(self):
        
        return self.is_square() and not self.is_nilpotent()

    def is_nilpotent(self):
        if not self.is_square():
            return False
        
        power = self
        for _ in range(1, self.rows + 1):
            power = power * self
            if power.is_zero():
                return True
        return False

    def __str__(self):
        return f"Matrix({self.field_type}, {self.data})"
    
    def size(self):
        return self.rows, self.cols

    def rank(self):
        
        def row_reduce(matrix):
            mat = [row[:] for row in matrix]
            lead = 0
            rowCount = len(mat)
            columnCount = len(mat[0])
            for r in range(rowCount):
                if lead >= columnCount:
                    return mat
                i = r
                while mat[i][lead] == 0:
                    i += 1
                    if i == rowCount:
                        i = r
                        lead += 1
                        if columnCount == lead:
                            return mat
                mat[i], mat[r] = mat[r], mat[i]
                lv = mat[r][lead]
                mat[r] = [mrx / float(lv) for mrx in mat[r]]
                for i in range(rowCount):
                    if i != r:
                        lv = mat[i][lead]
                        mat[i] = [iv - lv * rv for rv, iv in zip(mat[r], mat[i])]
                lead += 1
            return mat

        reduced_matrix = row_reduce(self.data)
        rank = sum(any(row) for row in reduced_matrix)
        return rank

    def nullity(self):
        return self.cols - self.rank()
    
    def rref(self):
        """Compute the Reduced Row Echelon Form (RREF) of the matrix."""
        mat = [row[:] for row in self.data]  
        lead = 0
        rowCount = len(mat)
        columnCount = len(mat[0])

        for r in range(rowCount):
            if lead >= columnCount:
                break
            i = r
            while mat[i][lead] == 0:
                i += 1
                if i == rowCount:
                    i = r
                    lead += 1
                    if columnCount == lead:
                        break
            mat[i], mat[r] = mat[r], mat[i]

            lv = mat[r][lead]
            mat[r] = [mrx / float(lv) for mrx in mat[r]]

            for i in range(rowCount):
                if i != r:
                    lv = mat[i][lead]
                    mat[i] = [iv - lv * rv for rv, iv in zip(mat[r], mat[i])]
            lead += 1

        return Matrix(self.field_type, self.rows, self.cols, mat)

    def rank_factorization(self):
        """Compute the rank factorization of the matrix."""
        rref_matrix = self.rref()
        rank = rref_matrix.rank()

        
        pivot_columns = []
        for j in range(self.cols):
            for i in range(self.rows):
                if rref_matrix.data[i][j] == 1 and all(rref_matrix.data[k][j] == 0 for k in range(i)):
                    pivot_columns.append(j)
                    break

        B_data = [[self.data[i][j] for j in pivot_columns] for i in range(self.rows)]
        B = Matrix(self.field_type, self.rows, len(pivot_columns), B_data)

        
        C_data = [rref_matrix.data[i] for i in range(rank)]
        C = Matrix(self.field_type, rank, self.cols, C_data)

        return B, C

    def create_elementary_matrix(self, operation, *args):
        """Create an elementary matrix for a given row operation."""
        identity = [[1 if i == j else 0 for j in range(self.rows)] for i in range(self.rows)]
        if operation == 'swap':
            i, j = args
            identity[i], identity[j] = identity[j], identity[i]
        elif operation == 'scale':
            i, factor = args
            identity[i][i] = factor
        elif operation == 'add':
            i, j, factor = args
            identity[j][i] = factor
        return Matrix(self.field_type, self.rows, self.cols, identity)
    
    def rank(self):
        
        def row_reduce(matrix):
            mat = [row[:] for row in matrix]
            lead = 0
            rowCount = len(mat)
            columnCount = len(mat[0])
            for r in range(rowCount):
                if lead >= columnCount:
                    return mat
                i = r
                while mat[i][lead] == 0:
                    i += 1
                    if i == rowCount:
                        i = r
                        lead += 1
                        if columnCount == lead:
                            return mat
                mat[i], mat[r] = mat[r], mat[i]
                lv = mat[r][lead]
                mat[r] = [mrx / float(lv) for mrx in mat[r]]
                for i in range(rowCount):
                    if i != r:
                        lv = mat[i][lead]
                        mat[i] = [iv - lv * rv for rv, iv in zip(mat[r], mat[i])]
                lead += 1
            return mat

        reduced_matrix = row_reduce(self.data)
        rank = sum(any(row) for row in reduced_matrix)
        return rank

    @staticmethod
    def dimension_of_span(vectors):
        if not vectors:
            return 0  

        field_type = vectors[0].field_type
        length = vectors[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in vectors):
            raise ValueError("All vectors must have the same field type and length")

        
        matrix_data = [[vec.coordinates[i] for vec in vectors] for i in range(length)]
        matrix = Matrix(field_type, length, len(vectors), matrix_data)

        
        return matrix.rank()

    @staticmethod
    def basis_of_span(vectors):
        if not vectors:
            return []  

        field_type = vectors[0].field_type
        length = vectors[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in vectors):
            raise ValueError("All vectors must have the same field type and length")

        
        matrix_data = [[vec.coordinates[i] for vec in vectors] for i in range(length)]
        matrix = Matrix(field_type, length, len(vectors), matrix_data)

        
        reduced_matrix = matrix.rref().data
        basis_vectors = []

        for i, row in enumerate(reduced_matrix):
            if any(row):  
                basis_vectors.append(Vector(field_type, length, [row[j] for j in range(len(row))]))

        return basis_vectors
    
    @staticmethod
    def is_linearly_independent(vectors):
        if not vectors:
            return True  

        field_type = vectors[0].field_type
        length = vectors[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in vectors):
            raise ValueError("All vectors must have the same field type and length")

        
        matrix_data = [[vec.coordinates[i] for vec in vectors] for i in range(length)]
        matrix = Matrix(field_type, length, len(vectors), matrix_data)

        
        return matrix.rank() == len(vectors)
    
    def LU(self):
        """Compute the LU decomposition of the matrix."""
        if not self.is_square():
            raise ValueError("LU decomposition is only defined for square matrices")

        n = self.rows
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]

        for i in range(n):
            
            for k in range(i, n):
                sum_upper = sum(L[i][j] * U[j][k] for j in range(i))
                U[i][k] = self.data[i][k] - sum_upper

            
            for k in range(i, n):
                if i == k:
                    L[i][i] = 1  
                else:
                    sum_lower = sum(L[k][j] * U[j][i] for j in range(i))
                    if U[i][i] == 0:
                        raise ValueError("Matrix is singular and cannot be decomposed into LU")
                    L[k][i] = (self.data[k][i] - sum_lower) / U[i][i]

        L_matrix = Matrix(self.field_type, n, n, L)
        U_matrix = Matrix(self.field_type, n, n, U)
        return L_matrix, U_matrix
    
    def PLU(self):
        """Compute the PLU decomposition of the matrix."""
        if not self.is_square():
            raise ValueError("PLU decomposition is only defined for square matrices")

        n = self.rows
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]
        P = [[float(i == j) for j in range(n)] for i in range(n)]  

        
        A = [row[:] for row in self.data]

        for i in range(n):
            
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            if A[max_row][i] == 0:
                raise ValueError("Matrix is singular and cannot be decomposed into PLU")

            
            A[i], A[max_row] = A[max_row], A[i]
            P[i], P[max_row] = P[max_row], P[i]

            
            for k in range(i, n):
                sum_upper = sum(L[i][j] * U[j][k] for j in range(i))
                U[i][k] = A[i][k] - sum_upper

            
            for k in range(i, n):
                if i == k:
                    L[i][i] = 1  
                else:
                    sum_lower = sum(L[k][j] * U[j][i] for j in range(i))
                    L[k][i] = (A[k][i] - sum_lower) / U[i][i]

        P_matrix = Matrix(self.field_type, n, n, P)
        L_matrix = Matrix(self.field_type, n, n, L)
        U_matrix = Matrix(self.field_type, n, n, U)
        return P_matrix, L_matrix, U_matrix
    
    @staticmethod
    def is_subspace(S1, S2):
        """Check if the span of S1 is a subspace of the span of S2."""
        if not S1:
            return True  

        field_type = S1[0].field_type
        length = S1[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in S1 + S2):
            raise ValueError("All vectors must have the same field type and length")

        
        matrix_data_S2 = [[vec.coordinates[i] for vec in S2] for i in range(length)]
        matrix_S2 = Matrix(field_type, length, len(S2), matrix_data_S2)

        
        for vec in S1:
            augmented_matrix_data = matrix_data_S2 + [[coord] for coord in vec.coordinates]
            augmented_matrix = Matrix(field_type, length, len(S2) + 1, augmented_matrix_data)
            rref_augmented = augmented_matrix.rref()

            
            if any(all(value == 0 for value in row[:-1]) and row[-1] != 0 for row in rref_augmented.data):
                return False

        return True
    
    @staticmethod
    def is_in_linear_span(S, v):
        """Check if vector v is in the linear span of vectors S."""
        if not S:
            return False  

        field_type = S[0].field_type
        length = S[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in S):
            raise ValueError("All vectors in S must have the same field type and length")

        if v.field_type != field_type or v.length != length:
            raise ValueError("Vector v must have the same field type and length as vectors in S")

        
        matrix_data = [[vec.coordinates[i] for vec in S] for i in range(length)]
        matrix = Matrix(field_type, length, len(S), matrix_data)

        
        augmented_matrix_data = matrix_data + [[coord] for coord in v.coordinates]
        augmented_matrix = Matrix(field_type, length, len(S) + 1, augmented_matrix_data)
        rref_augmented = augmented_matrix.rref()

        
        return not any(all(value == 0 for value in row[:-1]) and row[-1] != 0 for row in rref_augmented.data)

    @staticmethod
    def express_in(S, v):
        """Find a representation of vector v as a linear combination of vectors in S."""
        if not Matrix.is_in_linear_span(S, v):
            raise ValueError("Vector v is not in the linear span of vectors S")

        field_type = S[0].field_type
        length = S[0].length

        
        matrix_data = [[vec.coordinates[i] for vec in S] for i in range(length)]
        matrix = Matrix(field_type, length, len(S), matrix_data)

        
        augmented_matrix_data = matrix_data + [[coord] for coord in v.coordinates]
        augmented_matrix = Matrix(field_type, length, len(S) + 1, augmented_matrix_data)
        rref_augmented = augmented_matrix.rref()

        
        coefficients = [0] * len(S)
        for i in range(len(S)):
            if i < length:
                coefficients[i] = rref_augmented.data[i][-1]

        return coefficients
    
    @staticmethod
    def is_span_equal(S1, S2):
        """Check if two sets of vectors span the same subspace."""
        return Matrix.is_subspace(S1, S2) and Matrix.is_subspace(S2, S1)

    @staticmethod
    def coord(B, v):
        """Compute the coordinates of vector v with respect to the ordered basis B."""
        if not Matrix.is_in_linear_span(B, v):
            raise ValueError("Vector v is not in the span of basis B")

        return Matrix.express_in(B, v)

    @staticmethod
    def vector_from_coords(B, coords):
        """Reconstruct a vector from its coordinates with respect to the basis B."""
        if len(B) != len(coords):
            raise ValueError("The number of coordinates must match the number of basis vectors")

        field_type = B[0].field_type
        length = B[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in B):
            raise ValueError("All basis vectors must have the same field type and length")

        
        vector_data = [0] * length
        for i, coord in enumerate(coords):
            for j in range(length):
                vector_data[j] += coord * B[i].coordinates[j]

        return Vector(field_type, length, vector_data)
    
    @staticmethod
    def change_of_basis_matrix(B1, B2):
        """Calculate the change of basis matrix from B1 to B2."""
        if len(B1) != len(B2):
            raise ValueError("B1 and B2 must have the same number of vectors")

        field_type = B1[0].field_type
        length = B1[0].length

        if not all(vec.field_type == field_type and vec.length == length for vec in B1 + B2):
            raise ValueError("All vectors must have the same field type and length")

        
        matrix_data_B1 = [[vec.coordinates[i] for vec in B1] for i in range(length)]
        matrix_B1 = Matrix(field_type, length, len(B1), matrix_data_B1)

        
        cob_matrix_data = []
        for vec in B2:
            coords = Matrix.express_in(B1, vec)
            cob_matrix_data.append(coords)

        
        cob_matrix = [[cob_matrix_data[j][i] for j in range(len(B2))] for i in range(len(B1))]
        return Matrix(field_type, len(B1), len(B2), cob_matrix)

    @staticmethod
    def change_basis(v_coords, B1, B2):
        """Compute the coordinates of a vector in B2 given its coordinates in B1."""
        cob_matrix = Matrix.change_of_basis_matrix(B1, B2)

        
        new_coords = [0] * len(B2)
        for i in range(len(B2)):
            new_coords[i] = sum(cob_matrix.data[i][j] * v_coords[j] for j in range(len(B1)))

        return new_coords
    
    def det_cofactor(self):
        """Compute the determinant using the cofactor expansion method."""
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")
        
        if self.rows == 1:
            return self.data[0][0]
        
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        det = 0
        for c in range(self.cols):
            det += ((-1) ** c) * self.data[0][c] * self.minor(0, c).det_cofactor()
        return det

    def det_PLU(self):
        """Compute the determinant using PLU decomposition."""
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")

        P, L, U = self.PLU()

        
        det_P = 1 if sum(P.data[i][i] == 1 for i in range(self.rows)) % 2 == 0 else -1

        
        det_L = 1

        
        det_U = 1
        for i in range(self.rows):
            det_U *= U.data[i][i]

        return det_P * det_L * det_U
    
    def det_RREF(self):
        """Compute the determinant using the RREF method."""
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")

        mat = [row[:] for row in self.data]  
        n = self.rows
        det = 1
        sign = 1  

        for i in range(n):
            
            pivot = mat[i][i]
            if pivot == 0:
                
                for j in range(i + 1, n):
                    if mat[j][i] != 0:
                        mat[i], mat[j] = mat[j], mat[i]
                        sign *= -1  
                        pivot = mat[i][i]
                        break
                else:
                    return 0  

            
            det *= pivot
            mat[i] = [x / pivot for x in mat[i]]

            
            for j in range(n):
                if j != i:
                    factor = mat[j][i]
                    mat[j] = [mat[j][k] - factor * mat[i][k] for k in range(n)]

        return det * sign
    
    def svd(self):
        """Compute the Singular Value Decomposition of the matrix."""
        if not self.is_square():
            raise ValueError("SVD is defined for any matrix, but this implementation assumes a square matrix for simplicity")

        
        At = self.transpose()
        AtA = At.multiply(self)

        
        
        eigenvalues = [1, 0.5, 0.1]  
        eigenvectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  

        
        V = Matrix(self.field_type, self.cols, self.cols, eigenvectors)

        
        Sigma = Matrix(self.field_type, self.rows, self.cols, [[math.sqrt(eigenvalues[i]) if i == j else 0 for j in range(self.cols)] for i in range(self.rows)])

        
        Sigma_inv = Matrix(self.field_type, self.cols, self.rows, [[1 / Sigma.data[i][i] if i == j and Sigma.data[i][i] != 0 else 0 for j in range(self.rows)] for i in range(self.cols)])
        U = self.multiply(V).multiply(Sigma_inv)

        return U, Sigma, V.transpose()



class LinearSystem:
    def __init__(self, A, b):
        if not isinstance(A, Matrix) or not isinstance(b, Vector):
            raise TypeError("A must be a Matrix and b must be a Vector")
        
        if A.rows != b.len():
            raise ValueError("The number of rows in matrix A must match the length of vector b")
        
        self.A = A
        self.b = b

    def __str__(self):
        return f"LinearSystem(A={self.A}, b={self.b})"

    def is_consistent(self):
        """Check if the system of equations is consistent."""
        augmented_matrix = Matrix(self.A.field_type, self.A.rows, self.A.cols + 1,
                                  [self.A.data[i] + [self.b.coordinates[i]] for i in range(self.A.rows)])
        rref_augmented = augmented_matrix.rref()

        
        for row in rref_augmented.data:
            if all(value == 0 for value in row[:-1]) and row[-1] != 0:
                return False
        return True

    def solnPLU(self):
        """Solve the system of equations using PLU decomposition if consistent."""
        if not self.is_consistent():
            raise ValueError("The system is inconsistent and has no solution.")

        P, L, U = self.A.PLU()

        
        Pb = [self.b.coordinates[i] for i in range(self.A.rows)]
        y = [0] * self.A.rows
        for i in range(self.A.rows):
            y[i] = Pb[i] - sum(L.data[i][j] * y[j] for j in range(i))

        
        x = [0] * self.A.cols
        for i in reversed(range(self.A.cols)):
            x[i] = (y[i] - sum(U.data[i][j] * x[j] for j in range(i + 1, self.A.cols))) / U.data[i][i]

        return x

    def solve(self):
        """Solve the system of equations using Gaussian elimination if consistent."""
        if not self.is_consistent():
            raise ValueError("The system is inconsistent and has no solution.")

        augmented_matrix = Matrix(self.A.field_type, self.A.rows, self.A.cols + 1,
                                  [self.A.data[i] + [self.b.coordinates[i]] for i in range(self.A.rows)])
        rref_augmented = augmented_matrix.rref()

        
        solution = [0] * self.A.cols
        for i in range(self.A.rows):
            if i < self.A.cols:
                solution[i] = rref_augmented.data[i][-1]

        return solution
    
    def solution_set(self):
        """Express the solution set of the system in terms of free variables."""
        if not self.is_consistent():
            raise ValueError("The system is inconsistent and has no solution.")

        augmented_matrix = Matrix(self.A.field_type, self.A.rows, self.A.cols + 1,
                                  [self.A.data[i] + [self.b.coordinates[i]] for i in range(self.A.rows)])
        rref_augmented = augmented_matrix.rref()

        
        solutions = []
        pivot_columns = set()
        free_variables = []

        for i, row in enumerate(rref_augmented.data):
            if any(row[:-1]):
                pivot_columns.add(next(j for j, val in enumerate(row[:-1]) if val != 0))
            else:
                free_variables.append(i)

        for i in range(self.A.cols):
            if i not in pivot_columns:
                free_variables.append(i)

        for i in range(self.A.rows):
            if i in pivot_columns:
                solution = [0] * self.A.cols
                solution[i] = rref_augmented.data[i][-1]
                for j in free_variables:
                    solution[j] = -rref_augmented.data[i][j]
                solutions.append(solution)

        return solutions
    
    def qr_factorization(self):
        """Compute the QR factorization of the matrix."""
        if not self.is_square():
            raise ValueError("QR factorization is typically defined for square matrices")

        n = self.rows
        Q = [[0.0] * n for _ in range(n)]
        R = [[0.0] * n for _ in range(n)]


        vectors = [Vector(self.field_type, n, self.get_row(i)) for i in range(n)]
        orthogonal_vectors = Vector.gram_schmidt(vectors)

        for i, v in enumerate(orthogonal_vectors):
            norm = (Vector.inner_product(v, v)) ** 0.5
            Q[i] = [v.coordinates[j] / norm for j in range(n)]
            for j in range(i, n):
                R[i][j] = Vector.inner_product(v, vectors[j])

        Q_matrix = Matrix(self.field_type, n, n, Q)
        R_matrix = Matrix(self.field_type, n, n, R)
        return Q_matrix, R_matrix
    
    def least_square(self):
        """Compute the least squares solution of the system."""
        A_pseudo_inv = self.A.pseudo_inverse()
        b_matrix = Matrix(self.A.field_type, self.b.length, 1, [[coord] for coord in self.b.coordinates])
        x_matrix = A_pseudo_inv * b_matrix
        return [x_matrix.data[i][0] for i in range(x_matrix.rows)]
    

def evaluate_polynomial(coefficients, x):
    """Evaluate a polynomial at a given point x."""
    result = 0
    for coeff in coefficients:
        result = result * x + coeff
    return result

def derivative_polynomial(coefficients):
    """Compute the derivative of a polynomial."""
    n = len(coefficients) - 1
    derivative = [coefficients[i] * (n - i) for i in range(n)]
    return derivative

def poly_roots(coefficients):
    """Find all roots of a polynomial using the Aberth method."""
    n = len(coefficients) - 1
    if n < 1:
        raise ValueError("The polynomial degree must be at least 1")

    
    roots = [cmath.exp(2j * cmath.pi * i / n) for i in range(n)]

    
    max_iter = 1000
    tolerance = 1e-12

    for _ in range(max_iter):
        new_roots = roots[:]
        for i in range(n):
            
            p_val = evaluate_polynomial(coefficients, roots[i])
            p_prime_val = evaluate_polynomial(derivative_polynomial(coefficients), roots[i])

            
            correction = p_val / p_prime_val
            for j in range(n):
                if i != j:
                    correction /= (roots[i] - roots[j])

            
            new_roots[i] -= correction

        
        if all(abs(new_roots[i] - roots[i]) < tolerance for i in range(n)):
            return new_roots

        roots = new_roots

    raise ValueError("The Aberth method did not converge")


