clean:
	rm -rf *Test

DenseMatrixTest:
	g++ -std=c++11 -o DenseMatrixTest DenseMatrixTest.cpp ../SparseMatrix.cpp ../SparseVector.cpp ../DenseMatrix.cpp ../DenseVector.cpp ../Utility.cpp;
	./DenseMatrixTest
	echo "\n\nTests for DenseMatrix completed successfully.\n\n"

SparseMatrixTest:
	g++ -std=c++11 -o SparseMatrixTest SparseMatrixTest.cpp ../SparseMatrix.cpp ../SparseVector.cpp ../DenseMatrix.cpp ../DenseVector.cpp ../Utility.cpp;
	./SparseMatrixTest
	echo "\n\nTests for SparseMatrix completed successfully.\n\n"

SparseVectorTest:
	g++ -std=c++11 -o SparseVectorTest SparseVectorTest.cpp ../SparseMatrix.cpp ../SparseVector.cpp ../DenseMatrix.cpp ../DenseVector.cpp ../Utility.cpp;
	./SparseVectorTest
	echo "\n\nTests for SparseVector completed successfully.\n\n"

DenseVectorTest:
	g++ -std=c++11 -o DenseVectorTest DenseVectorTest.cpp ../SparseMatrix.cpp ../SparseVector.cpp ../DenseMatrix.cpp ../DenseVector.cpp ../Utility.cpp;
	./DenseVectorTest
	echo "\n\nTests for DenseVector completed successfully.\n\n"

UtilityTest:
	g++ -std=c++11 -o UtilityTest UtilityTest.cpp ../Utility.cpp;
	./UtilityTest
	echo "\n\nTests for Utility completed successfully.\n\n"


test: DenseMatrixTest SparseMatrixTest SparseVectorTest DenseVectorTest UtilityTest
