from numpy import array,mat,transpose,zeros,identity,eye,diag,shape,ones 
from numpy.linalg import inv,det,eig,matrix_rank
from scipy.linalg import lu

# A and B matrix defined using keyword "array" 
A=array([[2,3,7],[4,1,11],[5,13,9]])
B=array([[5,8,11],[3,7,4],[13,2,1]])

# Y and Z matrix defined using keyword "mat" (Elements same as in A and B) 
Y=mat([[2, 3, 7],[4, 1, 11],[5,13,9]])
Z=mat([[5,8,11],[3,7,4],[13,2,1]])

print("A = \n",A,"\n\nB= \n",B) 
print("\nY= \n",Y,"\n\nZ= \n",Z)

# Size of matrix for both cases (matrix defined by array and mat)
m,n=shape(A); m1,n1=shape(Z); # m and n indicates row and column respectively

print("\nm=",m,"\n\nn=",n) 
print("\nm1=",m1,"\n\nn1=",n1)
 
# Addition & Subtraction of matrix for both cases (matrix defined by array and mat) 
C=A+B; D=A-B;
C1=Y+Z; D1=Y-Z;

print("\nAddition=\n",C,"\n\nAddition1=\n",C1) 
print("\nSubtraction=\n",D,"\n\nSubtraction1=\n",D1)

# Multiplication of matrix for both cases (matrix defined by array and mat) 
E=A*B; F=A.dot(B);
E1=Y*Z; F1=Y.dot(Z);

print("\nMul1=\n",E,"\n\nMul11=\n",E1) 
print("\nMul2=\n",F,"\n\nMul21=\n",F1)

# Inverse, transpose and determinant of matrix for both cases (matrix defined by array and mat)
G=inv(A);G1=inv(Y);G2=Y**(-1);
H=transpose(A);H1=transpose(Y); H2=round(det(A)); H3=round(det(Y))

print("\nInverse=\n",G,"\n\nInverse1=\n",G1,"\n\nInverse2=\n",G2) 
print("\nTranspose=\n",H,"\n\nTranspose1=\n",H1)
print("\nDeterminant=\n",H2,"\n\nDeterminant1=\n",H3)

# General keywords for getting all zero or one or identity matrix or diagonal matrix 
I=identity(3);I1=eye(3); # identity matrix of size 3*3
J=ones([2,3])	# matrix of size 2*3 having all elements as 1 
K=zeros([3,3])	# matrix of size 3*3 having all elements as 0 
P=diag([4,3,2])	# diagonal matrix

print("\nI=\n",I,"\n\nI1=\n",I1) 
print("\nJ=\n",J,"\n\nK=\n",K) 
print("\nP=\n",P)

# Rank,Eigen value & Eigen Vectors of matrix for both cases (matrix defined by array and # mat)
R=matrix_rank(A); R1=matrix_rank(Y) 
Val,Vect=eig(A); Val1,Vect1=eig(Y)

print("\nRank=\n",R,"\n\nRank1=\n",R1)
print("\nEigen value=\n",Val,"\n\nEigen Vectors=\n",Vect) 
print("\nEigen value1=\n",Val1,"\n\nEigen Vectors1=\n",Vect1)

# Lower & Upper triangle of the matrix 
L=lu(A)[1]; 
L1=lu(Y)[1]
U=lu(A)[2]; U1=lu(Y)[2]

print("\nL=\n",L,"\n\nL1=\n",L1)
print("\nU=\n",U,"\n\nU1=\n",U1)