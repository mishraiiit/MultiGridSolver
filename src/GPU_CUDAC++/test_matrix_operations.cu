/*
 * Test file for matrix operations
 * Tests: 
 * 1) Reading SmallTestMatrix.mtx and printing it
 * 2) Copying matrix to GPU
 * 3) Copying matrix back to CPU and printing again
 * 
 * @author: Test suite
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <string>
#include <cmath>
#include "MatrixIO.cu"
#include "MatrixOperations.cu"
#include "GPUDebug.cu"
#include "MatrixAccess.cu"

int main() {
    // Step 1: Read SmallTestMatrix.mtx from file to CPU memory
    std::string filename = "../../matrices/SmallTestMatrix.mtx";
    
    printLines();
    printInfo("Starting Matrix Operations Test", 0);
    printLines();
    
    printInfo("Step 1: Reading matrix from file to CPU memory", 0);
    MatrixCSR* matrix_cpu = readMatrixCPUMemoryCSR(filename);
    
    // Print matrix details
    printInfo("Matrix successfully read!", 4);
    printInfo("Matrix dimensions: " + itoa(matrix_cpu->rows) + " x " + itoa(matrix_cpu->cols), 4);
    printInfo("Number of non-zero entries: " + itoa(matrix_cpu->nnz), 4);
    
    // Step 2: Print the matrix to confirm it's correct
    printLines();
    printInfo("Step 2: Printing matrix in COO format (row col value):", 0);
    printCSRCPU(matrix_cpu);
    
    // Step 3: Copy matrix to GPU
    printLines();
    printInfo("Step 3: Copying matrix from CPU to GPU", 0);
    MatrixCSR* matrix_gpu = deepCopyMatrixCSRCPUtoGPU(matrix_cpu);
    printInfo("Matrix successfully copied to GPU!", 4);
    
    // Step 4: Copy matrix back from GPU to CPU
    printLines();
    printInfo("Step 4: Copying matrix back from GPU to CPU", 0);
    MatrixCSR* matrix_cpu_copy = deepCopyMatrixCSRGPUtoCPU(matrix_gpu);
    printInfo("Matrix successfully copied back to CPU!", 4);
    
    // Step 5: Print the matrix again to verify
    printLines();
    printInfo("Step 5: Printing matrix again after GPU round-trip:", 0);
    printCSRCPU(matrix_cpu_copy);
    
    // Verify that the matrices are identical
    printLines();
    printInfo("Verification: Checking if matrices are identical", 0);
    
    bool identical = true;
    
    // Check dimensions
    if (matrix_cpu->rows != matrix_cpu_copy->rows || 
        matrix_cpu->cols != matrix_cpu_copy->cols ||
        matrix_cpu->nnz != matrix_cpu_copy->nnz) {
        identical = false;
        printInfo("ERROR: Matrix dimensions don't match!", 4);
    }
    
    // Check row pointers
    if (identical) {
        for (int i = 0; i <= matrix_cpu->rows; i++) {
            if (matrix_cpu->i[i] != matrix_cpu_copy->i[i]) {
                identical = false;
                printInfo("ERROR: Row pointers don't match at index " + itoa(i), 4);
                break;
            }
        }
    }
    
    // Check column indices and values
    if (identical) {
        for (int i = 0; i < matrix_cpu->nnz; i++) {
            if (matrix_cpu->j[i] != matrix_cpu_copy->j[i] ||
                matrix_cpu->val[i] != matrix_cpu_copy->val[i]) {
                identical = false;
                printInfo("ERROR: Column indices or values don't match at index " + itoa(i), 4);
                break;
            }
        }
    }
    
    if (identical) {
        printInfo("SUCCESS: Matrices are identical! CPU->GPU->CPU CSR copy work correctly.", 4);
    } else {
        printInfo("FAILURE: Matrices are different after GPU operations.", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Test getElementMatrixCSR function
    printLines();
    printInfo("Testing getElementMatrixCSR function", 0);
    printInfo("Checking individual matrix elements (0-indexed):", 4);
    
    // Test structure to check various elements
    struct TestCase {
        int row;
        int col;
        float expected;
        const char* description;
    };
    
    // Define test cases based on the matrix data
    // Note: Matrix file uses 1-based indexing, but CSR uses 0-based
    TestCase testCases[] = {
        // Non-zero elements from the matrix (sequential values 1-17)
        {0, 0, 1.0f, "Element (0,0) - first element"},
        {0, 2, 2.0f, "Element (0,2) - value 2"},
        {0, 3, 3.0f, "Element (0,3) - value 3"},
        {0, 7, 4.0f, "Element (0,7) - value 4"},
        {1, 3, 5.0f, "Element (1,3) - value 5"},
        {1, 4, 6.0f, "Element (1,4) - value 6"},
        {2, 0, 7.0f, "Element (2,0) - value 7"},
        {2, 1, 8.0f, "Element (2,1) - value 8"},
        {2, 4, 9.0f, "Element (2,4) - value 9"},
        {3, 3, 10.0f, "Element (3,3) - diagonal, value 10"},
        {3, 4, 11.0f, "Element (3,4) - value 11"},
        {4, 0, 12.0f, "Element (4,0) - value 12"},
        {4, 1, 13.0f, "Element (4,1) - value 13"},
        {4, 2, 14.0f, "Element (4,2) - value 14"},
        {4, 3, 15.0f, "Element (4,3) - value 15"},
        {4, 4, 16.0f, "Element (4,4) - diagonal, value 16"},
        {6, 5, 17.0f, "Element (6,5) - value 17"},
        
        // Zero elements (not stored in sparse format)
        {0, 1, 0.0f, "Element (0,1) - zero"},
        {1, 0, 0.0f, "Element (1,0) - zero"},
        {1, 1, 0.0f, "Element (1,1) - zero diagonal"},
        {5, 0, 0.0f, "Element (5,0) - zero in empty row"},
        {5, 5, 0.0f, "Element (5,5) - zero in empty row"},
        {8, 6, 0.0f, "Element (8,6) - last row, last col"},
        {7, 2, 0.0f, "Element (7,2) - zero in sparse row"}
    };
    
    int numTests = sizeof(testCases) / sizeof(TestCase);
    int passed = 0;
    int failed = 0;
    
    for (int i = 0; i < numTests; i++) {
        float value = getElementMatrixCSR(matrix_cpu, testCases[i].row, testCases[i].col);
        
        if (value == testCases[i].expected) {
            passed++;
            if (value != 0.0f) {
                printInfo("✓ " + std::string(testCases[i].description) + 
                         " = " + std::to_string(value), 8);
            }
        } else {
            failed++;
            printInfo("✗ " + std::string(testCases[i].description) + 
                     " - Expected: " + std::to_string(testCases[i].expected) + 
                     ", Got: " + std::to_string(value), 8);
        }
    }
    
    printInfo("Element access test results: " + itoa(passed) + " passed, " + 
              itoa(failed) + " failed out of " + itoa(numTests) + " tests", 4);
    
    if (failed > 0) {
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Test boundary conditions
    printLines();
    printInfo("Testing boundary conditions", 0);
    
    // First element
    float firstElem = getElementMatrixCSR(matrix_cpu, 0, 0);
    printInfo("First element (0,0) = " + std::to_string(firstElem), 4);
    
    // Last possible element
    float lastElem = getElementMatrixCSR(matrix_cpu, matrix_cpu->rows - 1, matrix_cpu->cols - 1);
    printInfo("Last element (" + itoa(matrix_cpu->rows - 1) + "," + 
              itoa(matrix_cpu->cols - 1) + ") = " + std::to_string(lastElem), 4);
    
    // Test entire first row
    printInfo("First row elements:", 4);
    std::string rowStr = "Row 0: ";
    for (int j = 0; j < matrix_cpu->cols; j++) {
        float val = getElementMatrixCSR(matrix_cpu, 0, j);
        rowStr += std::to_string(val) + " ";
    }
    printInfo(rowStr, 8);
    
    // Test entire fifth row (row index 4) which has many non-zeros
    printInfo("Fifth row elements (row index 4):", 4);
    rowStr = "Row 4: ";
    for (int j = 0; j < matrix_cpu->cols; j++) {
        float val = getElementMatrixCSR(matrix_cpu, 4, j);
        rowStr += std::to_string(val) + " ";
    }
    printInfo(rowStr, 8);
    
    // Test CSR vs CSC consistency
    printLines();
    printInfo("Testing CSR vs CSC consistency using GPU conversion", 0);
    
    // Create cuSPARSE handle for conversion
    cusparseHandle_t cusparse_handle;
    cusparseStatus_t status = cusparseCreate(&cusparse_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printInfo("ERROR: Failed to create cuSPARSE handle!", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    printInfo("Created cuSPARSE handle", 4);
    
    // Step 1: Copy CSR matrix to GPU (reuse existing GPU matrix)
    printInfo("Using existing CSR matrix on GPU", 4);
    
    // Step 2: Convert CSR GPU to CSC GPU
    printInfo("Converting CSR to CSC on GPU", 4);
    MatrixCSC* matrix_csc_gpu = convertCSRGPU_cudaSparse(matrix_gpu, cusparse_handle);
    printInfo("Conversion complete", 8);
    
    // Step 3: Copy CSC GPU back to CPU
    printInfo("Copying CSC matrix from GPU to CPU", 4);
    MatrixCSC* matrix_csc_cpu = deepCopyMatrixCSCGPUtoCPU(matrix_csc_gpu);
    printInfo("CSC matrix successfully copied to CPU", 8);
    
    // Compare dimensions
    printInfo("Comparing dimensions:", 4);
    printInfo("CSR dimensions: " + itoa(matrix_cpu->rows) + " x " + itoa(matrix_cpu->cols), 8);
    printInfo("CSC dimensions: " + itoa(matrix_csc_cpu->rows) + " x " + itoa(matrix_csc_cpu->cols), 8);
    printInfo("CSR nnz: " + itoa(matrix_cpu->nnz), 8);
    printInfo("CSC nnz: " + itoa(matrix_csc_cpu->nnz), 8);
    
    bool dimensions_match = (matrix_cpu->rows == matrix_csc_cpu->rows) &&
                           (matrix_cpu->cols == matrix_csc_cpu->cols) &&
                           (matrix_cpu->nnz == matrix_csc_cpu->nnz);
    
    if (dimensions_match) {
        printInfo("✓ Dimensions match between CSR and CSC formats", 8);
    } else {
        printInfo("✗ Dimensions mismatch between CSR and CSC formats!", 8);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Test specific elements in both formats
    printInfo("Comparing element values between CSR and CSC:", 4);
    
    struct CompareTest {
        int row;
        int col;
        const char* description;
    };
    
    // Test cases including (0,7) which is (1,8) in 1-based indexing
    CompareTest compareTests[] = {
        {0, 7, "(0,7) = (1,8 in file) - value 4"},
        {0, 0, "(0,0) - first element, value 1"},
        {0, 2, "(0,2) - value 2"},
        {0, 3, "(0,3) - value 3"},
        {1, 3, "(1,3) - value 5"},
        {1, 4, "(1,4) - value 6"},
        {2, 0, "(2,0) - value 7"},
        {2, 1, "(2,1) - value 8"},
        {2, 4, "(2,4) - value 9"},
        {4, 1, "(4,1) - value 13"},
        {4, 4, "(4,4) - diagonal element, value 16"},
        {6, 5, "(6,5) - value 17"},
        {8, 9, "(8,9) - last element (should be 0)"},
        {5, 5, "(5,5) - zero element in empty row"},
        {0, 1, "(0,1) - zero element"},
        {7, 7, "(7,7) - zero diagonal element"}
    };
    
    int numCompareTests = sizeof(compareTests) / sizeof(CompareTest);
    int csr_csc_matches = 0;
    int csr_csc_mismatches = 0;
    
    for (int i = 0; i < numCompareTests; i++) {
        float csr_val = getElementMatrixCSR(matrix_cpu, compareTests[i].row, compareTests[i].col);
        float csc_val = getElementMatrixCSC(matrix_csc_cpu, compareTests[i].row, compareTests[i].col);
        
        if (csr_val == csc_val) {
            csr_csc_matches++;
            if (csr_val != 0.0f || i < 5) {  // Show non-zero values and first few tests
                printInfo("✓ " + std::string(compareTests[i].description) + 
                         ": CSR=" + std::to_string(csr_val) + 
                         ", CSC=" + std::to_string(csc_val), 8);
            }
        } else {
            csr_csc_mismatches++;
            printInfo("✗ " + std::string(compareTests[i].description) + 
                     ": CSR=" + std::to_string(csr_val) + 
                     ", CSC=" + std::to_string(csc_val) + " MISMATCH!", 8);
        }
    }
    
    printInfo("CSR vs CSC comparison: " + itoa(csr_csc_matches) + " matches, " + 
              itoa(csr_csc_mismatches) + " mismatches out of " + 
              itoa(numCompareTests) + " tests", 4);
    
    if (csr_csc_mismatches > 0) {
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Exhaustive test for SmallTestMatrix values
    printLines();
    printInfo("EXHAUSTIVE TEST: Verifying all SmallTestMatrix values across formats", 0);
    printLines();
    
    // Define all expected non-zero values from SmallTestMatrix.mtx
    // Format: (row-1, col-1, value) - converting from 1-based to 0-based indexing
    struct MatrixEntry {
        int row;
        int col;
        float value;
    };
    
    MatrixEntry expectedEntries[] = {
        {0, 0, 1.0f},     // 1 1 1
        {0, 2, 2.0f},     // 1 3 2
        {0, 3, 3.0f},     // 1 4 3
        {0, 7, 4.0f},     // 1 8 4
        {1, 3, 5.0f},     // 2 4 5
        {1, 4, 6.0f},     // 2 5 6
        {2, 0, 7.0f},     // 3 1 7
        {2, 1, 8.0f},     // 3 2 8
        {2, 4, 9.0f},     // 3 5 9
        {3, 3, 10.0f},    // 4 4 10
        {3, 4, 11.0f},    // 4 5 11
        {4, 0, 12.0f},    // 5 1 12
        {4, 1, 13.0f},    // 5 2 13
        {4, 2, 14.0f},    // 5 3 14
        {4, 3, 15.0f},    // 5 4 15
        {4, 4, 16.0f},    // 5 5 16
        {6, 5, 17.0f}     // 7 6 17
    };
    
    int numExpectedEntries = sizeof(expectedEntries) / sizeof(MatrixEntry);
    
    // Test 1: Verify CSR CPU values
    printInfo("Test 1: Verifying all non-zero values in CSR CPU format", 0);
    bool csr_cpu_correct = true;
    
    for (int i = 0; i < numExpectedEntries; i++) {
        float val = getElementMatrixCSR(matrix_cpu, expectedEntries[i].row, expectedEntries[i].col);
        if (val != expectedEntries[i].value) {
            csr_cpu_correct = false;
            printInfo("✗ CSR CPU mismatch at (" + itoa(expectedEntries[i].row) + "," + 
                     itoa(expectedEntries[i].col) + "): expected " + 
                     std::to_string(expectedEntries[i].value) + ", got " + 
                     std::to_string(val), 8);
        } else {
            printInfo("✓ CSR CPU (" + itoa(expectedEntries[i].row) + "," + 
                     itoa(expectedEntries[i].col) + ") = " + 
                     std::to_string(val), 8);
        }
    }
    
    if (csr_cpu_correct) {
        printInfo("✓ SUCCESS: All CSR CPU values match expected values!", 4);
    } else {
        printInfo("✗ FAILURE: CSR CPU values don't match!", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Test 2: Verify CSR GPU values (need to copy matrix element-by-element from GPU)
    printInfo("\nTest 2: Verifying all non-zero values in CSR GPU format", 0);
    
    // We'll use the already copied back matrix (matrix_cpu_copy) which came from GPU
    bool csr_gpu_correct = true;
    
    for (int i = 0; i < numExpectedEntries; i++) {
        float val = getElementMatrixCSR(matrix_cpu_copy, expectedEntries[i].row, expectedEntries[i].col);
        if (val != expectedEntries[i].value) {
            csr_gpu_correct = false;
            printInfo("✗ CSR GPU->CPU mismatch at (" + itoa(expectedEntries[i].row) + "," + 
                     itoa(expectedEntries[i].col) + "): expected " + 
                     std::to_string(expectedEntries[i].value) + ", got " + 
                     std::to_string(val), 8);
        } else {
            printInfo("✓ CSR GPU->CPU (" + itoa(expectedEntries[i].row) + "," + 
                     itoa(expectedEntries[i].col) + ") = " + 
                     std::to_string(val), 8);
        }
    }
    
    if (csr_gpu_correct) {
        printInfo("✓ SUCCESS: All CSR GPU values match expected values after round-trip!", 4);
    } else {
        printInfo("✗ FAILURE: CSR GPU values don't match after round-trip!", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Test 3: Verify CSC GPU values (through CPU copy)
    printInfo("\nTest 3: Verifying all non-zero values in CSC format (GPU->CPU)", 0);
    bool csc_correct = true;
    
    for (int i = 0; i < numExpectedEntries; i++) {
        float val = getElementMatrixCSC(matrix_csc_cpu, expectedEntries[i].row, expectedEntries[i].col);
        if (val != expectedEntries[i].value) {
            csc_correct = false;
            printInfo("✗ CSC mismatch at (" + itoa(expectedEntries[i].row) + "," + 
                     itoa(expectedEntries[i].col) + "): expected " + 
                     std::to_string(expectedEntries[i].value) + ", got " + 
                     std::to_string(val), 8);
        } else {
            printInfo("✓ CSC (" + itoa(expectedEntries[i].row) + "," + 
                     itoa(expectedEntries[i].col) + ") = " + 
                     std::to_string(val), 8);
        }
    }
    
    if (csc_correct) {
        printInfo("✓ SUCCESS: All CSC values match expected values!", 4);
    } else {
        printInfo("✗ FAILURE: CSC values don't match!", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Test 4: Verify all zero values are actually zero
    printInfo("\nTest 4: Verifying zero values in sparse matrix", 0);
    bool zeros_correct = true;
    int zero_count = 0;
    
    // Check all positions that should be zero
    for (int row = 0; row < matrix_cpu->rows; row++) {
        for (int col = 0; col < matrix_cpu->cols; col++) {
            // Check if this position is in our expected entries
            bool is_nonzero = false;
            for (int k = 0; k < numExpectedEntries; k++) {
                if (expectedEntries[k].row == row && expectedEntries[k].col == col) {
                    is_nonzero = true;
                    break;
                }
            }
            
            if (!is_nonzero) {
                // This should be zero
                float csr_val = getElementMatrixCSR(matrix_cpu, row, col);
                float csc_val = getElementMatrixCSC(matrix_csc_cpu, row, col);
                
                if (csr_val != 0.0f || csc_val != 0.0f) {
                    zeros_correct = false;
                    printInfo("✗ Expected zero at (" + itoa(row) + "," + itoa(col) + 
                             ") but got CSR=" + std::to_string(csr_val) + 
                             ", CSC=" + std::to_string(csc_val), 8);
                }
                zero_count++;
            }
        }
    }
    
    printInfo("Checked " + itoa(zero_count) + " zero positions", 4);
    if (zeros_correct) {
        printInfo("✓ SUCCESS: All zero values are correctly represented!", 4);
    } else {
        printInfo("✗ FAILURE: Some positions that should be zero are non-zero!", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Summary
    printLines();
    printInfo("EXHAUSTIVE TEST SUMMARY:", 0);
    printInfo("All exhaustive tests PASSED!", 4);
    
    // Test GPU debug kernels
    printLines();
    printInfo("Testing GPU Debug Kernels", 0);
    printLines();
    
    // Test debugCSR kernel
    printInfo("Test 5: Printing entire matrix using debugCSR kernel on GPU", 0);
    printInfo("Matrix in dense format (CSR):", 4);
    debugCSR<<<1,1>>>(matrix_gpu);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printInfo("✗ CUDA Error in debugCSR: " + std::string(cudaGetErrorString(cudaErr)), 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    } else {
        printInfo("✓ debugCSR kernel executed successfully", 4);
    }
    
    // Test debugCSC kernel
    printInfo("\nTest 6: Printing entire matrix using debugCSC kernel on GPU", 0);
    printInfo("Matrix in dense format (CSC):", 4);
    debugCSC<<<1,1>>>(matrix_csc_gpu);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printInfo("✗ CUDA Error in debugCSC: " + std::string(cudaGetErrorString(cudaErr)), 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    } else {
        printInfo("✓ debugCSC kernel executed successfully", 4);
    }
    
    // Compare outputs (manually verify they are the same)
    printInfo("\nNote: Both CSR and CSC debug outputs should show the same matrix in dense format", 4);
    printInfo("Expected 9x10 matrix with 17 non-zero values (1-17) and rest zeros", 4);
    
    // Test transpose function
    printLines();
    printInfo("Testing transposeCSRGPU_cudaSparse", 0);
    printLines();
    
    printInfo("Test 7: Transposing CSR matrix on GPU and verifying correctness", 0);
    
    // Transpose the CSR matrix on GPU
    printInfo("Transposing CSR matrix on GPU...", 4);
    MatrixCSR* matrix_transpose_gpu = transposeCSRGPU_cudaSparse(matrix_gpu, cusparse_handle);
    printInfo("Transpose operation complete", 8);
    
    // Copy transposed matrix back to CPU
    printInfo("Copying transposed matrix from GPU to CPU...", 4);
    MatrixCSR* matrix_transpose_cpu = deepCopyMatrixCSRGPUtoCPU(matrix_transpose_gpu);
    printInfo("Transposed matrix successfully copied to CPU", 8);
    
    // Verify dimensions
    printInfo("Verifying transposed matrix dimensions:", 4);
    printInfo("Original matrix: " + itoa(matrix_cpu->rows) + " x " + itoa(matrix_cpu->cols), 8);
    printInfo("Transposed matrix: " + itoa(matrix_transpose_cpu->rows) + " x " + itoa(matrix_transpose_cpu->cols), 8);
    
    bool dimensions_correct = (matrix_cpu->rows == matrix_transpose_cpu->cols) && 
                             (matrix_cpu->cols == matrix_transpose_cpu->rows);
    
    if (dimensions_correct) {
        printInfo("✓ Transposed matrix dimensions are correct", 8);
    } else {
        printInfo("✗ Transposed matrix dimensions are incorrect!", 8);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Exhaustive element-by-element comparison
    printInfo("\nPerforming exhaustive element comparison...", 4);
    int total_elements_checked = matrix_cpu->rows * matrix_cpu->cols;
    int transpose_matches = 0;
    int transpose_mismatches = 0;
    
    // Compare all elements: original[i,j] should equal transposed[j,i]
    for (int i = 0; i < matrix_cpu->rows; i++) {
        for (int j = 0; j < matrix_cpu->cols; j++) {
            float original_val = getElementMatrixCSR(matrix_cpu, i, j);
            float transposed_val = getElementMatrixCSR(matrix_transpose_cpu, j, i);
            
            if (original_val == transposed_val) {
                transpose_matches++;
                // Print first few non-zero matches for verification
                if (original_val != 0.0f && transpose_matches <= 10) {
                    printInfo("✓ (" + itoa(i) + "," + itoa(j) + ") = " + std::to_string(original_val) + 
                             " matches (" + itoa(j) + "," + itoa(i) + ") = " + std::to_string(transposed_val), 8);
                }
            } else {
                transpose_mismatches++;
                // Print all mismatches (should be none)
                printInfo("✗ Mismatch: (" + itoa(i) + "," + itoa(j) + ") = " + std::to_string(original_val) + 
                         " but (" + itoa(j) + "," + itoa(i) + ") = " + std::to_string(transposed_val), 8);
            }
        }
    }
    
    printInfo("Transpose verification complete:", 4);
    printInfo("Total elements checked: " + itoa(total_elements_checked), 8);
    printInfo("Matches: " + itoa(transpose_matches), 8);
    printInfo("Mismatches: " + itoa(transpose_mismatches), 8);
    
    if (transpose_mismatches == 0 && dimensions_correct) {
        printInfo("✓ SUCCESS: Matrix transpose is correct! All elements match perfectly.", 4);
    } else {
        printInfo("✗ FAILURE: Matrix transpose has errors!", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Print a few rows of the transposed matrix for visual verification
    printInfo("\nSample of transposed matrix (first 3 rows):", 4);
    for (int i = 0; i < std::min(3, matrix_transpose_cpu->rows); i++) {
        std::string rowStr = "Row " + itoa(i) + ": ";
        for (int j = 0; j < std::min(10, matrix_transpose_cpu->cols); j++) {
            float val = getElementMatrixCSR(matrix_transpose_cpu, i, j);
            rowStr += std::to_string(val) + " ";
        }
        printInfo(rowStr, 8);
    }
    
    // Test sparse matrix multiplication
    printLines();
    printInfo("Testing spmatrixmult_cudaSparse", 0);
    printLines();
    
    printInfo("Test 8: Multiplying matrix with its transpose (A * A^T)", 0);
    printInfo("Expected result: 9x9 matrix", 4);
    
    // Multiply matrix with its transpose on GPU
    printInfo("Performing sparse matrix multiplication on GPU...", 4);
    MatrixCSR* matrix_product_gpu = spmatrixmult_cudaSparse(matrix_gpu, matrix_transpose_gpu, cusparse_handle);
    printInfo("Multiplication complete", 8);
    
    // Copy result back to CPU
    printInfo("Copying product matrix from GPU to CPU...", 4);
    MatrixCSR* matrix_product_cpu = deepCopyMatrixCSRGPUtoCPU(matrix_product_gpu);
    printInfo("Product matrix successfully copied to CPU", 8);
    
    // Verify dimensions
    printInfo("Verifying product matrix dimensions:", 4);
    printInfo("Product matrix: " + itoa(matrix_product_cpu->rows) + " x " + itoa(matrix_product_cpu->cols), 8);
    printInfo("Number of non-zeros: " + itoa(matrix_product_cpu->nnz), 8);
    
    bool product_dimensions_correct = (matrix_product_cpu->rows == 9) && (matrix_product_cpu->cols == 9);
    
    if (product_dimensions_correct) {
        printInfo("✓ Product matrix dimensions are correct (9x9)", 8);
    } else {
        printInfo("✗ Product matrix dimensions are incorrect!", 8);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Compute expected values using serial implementation
    printInfo("\nComputing expected values using serial implementation...", 4);
    
    // For verification, compute A * A^T manually for specific elements
    // Each element (i,j) of the product is the dot product of row i of A with row j of A
    struct ProductTest {
        int row;
        int col;
        float expected;
        const char* description;
    };
    
    // Calculate some expected values manually
    // Row 0 of A has non-zeros at positions: (0,1), (2,2), (3,3), (7,4)
    // Row 0 dot Row 0 = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    ProductTest productTests[] = {
        {0, 0, 30.0f, "(0,0) - diagonal element"},
        {1, 1, 61.0f, "(1,1) - diagonal element"}, // 5*5 + 6*6 = 25 + 36 = 61
        {2, 2, 194.0f, "(2,2) - diagonal element"}, // 7*7 + 8*8 + 9*9 = 49 + 64 + 81 = 194
        {3, 3, 221.0f, "(3,3) - diagonal element"}, // 10*10 + 11*11 = 100 + 121 = 221
        {4, 4, 750.0f, "(4,4) - diagonal element"}, // 12*12 + 13*13 + 14*14 + 15*15 + 16*16 = 144 + 169 + 196 + 225 + 256 = 990
        {6, 6, 289.0f, "(6,6) - diagonal element"}, // 17*17 = 289
        {0, 1, 33.0f, "(0,1) - off-diagonal"}, // Row 0 dot Row 1 = 3*5 + 4*6 = 15 + 24 = 39
        {1, 0, 33.0f, "(1,0) - symmetric to (0,1)"},
        {0, 2, 48.0f, "(0,2) - off-diagonal"}, // Row 0 dot Row 2 = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
        {2, 0, 48.0f, "(2,0) - symmetric to (0,2)"}
    };
    
    // First, let's compute the correct expected values by doing the multiplication manually
    printInfo("Computing all product values manually...", 4);
    
    // Create a dense representation for easier computation
    float** dense_A = new float*[9];
    for (int i = 0; i < 9; i++) {
        dense_A[i] = new float[10];
        for (int j = 0; j < 10; j++) {
            dense_A[i][j] = getElementMatrixCSR(matrix_cpu, i, j);
        }
    }
    
    // Compute A * A^T manually
    float** expected_product = new float*[9];
    for (int i = 0; i < 9; i++) {
        expected_product[i] = new float[9];
        for (int j = 0; j < 9; j++) {
            expected_product[i][j] = 0.0f;
            // Dot product of row i with row j
            for (int k = 0; k < 10; k++) {
                expected_product[i][j] += dense_A[i][k] * dense_A[j][k];
            }
        }
    }
    
    // Update expected values based on actual computation
    productTests[0].expected = expected_product[0][0];
    productTests[1].expected = expected_product[1][1];
    productTests[2].expected = expected_product[2][2];
    productTests[3].expected = expected_product[3][3];
    productTests[4].expected = expected_product[4][4];
    productTests[5].expected = expected_product[6][6];
    productTests[6].expected = expected_product[0][1];
    productTests[7].expected = expected_product[1][0];
    productTests[8].expected = expected_product[0][2];
    productTests[9].expected = expected_product[2][0];
    
    // Test specific elements
    printInfo("Testing specific product elements:", 4);
    int numProductTests = sizeof(productTests) / sizeof(ProductTest);
    bool product_values_correct = true;
    
    for (int i = 0; i < numProductTests; i++) {
        float actual = getElementMatrixCSR(matrix_product_cpu, productTests[i].row, productTests[i].col);
        if (actual == productTests[i].expected) {
            printInfo("✓ " + std::string(productTests[i].description) + " = " + 
                     std::to_string(actual), 8);
        } else {
            printInfo("✗ " + std::string(productTests[i].description) + 
                     " - Expected: " + std::to_string(productTests[i].expected) + 
                     ", Got: " + std::to_string(actual), 8);
            product_values_correct = false;
        }
    }
    
    // Exhaustive comparison
    printInfo("\nPerforming exhaustive comparison of all elements...", 4);
    int product_matches = 0;
    int product_mismatches = 0;
    
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            float expected = expected_product[i][j];
            float actual = getElementMatrixCSR(matrix_product_cpu, i, j);
            
            if (std::abs(expected - actual) < 1e-5) { // Use epsilon for float comparison
                product_matches++;
                if (expected != 0.0f && product_matches <= 10) {
                    printInfo("✓ (" + itoa(i) + "," + itoa(j) + ") = " + 
                             std::to_string(actual), 8);
                }
            } else {
                product_mismatches++;
                printInfo("✗ Mismatch at (" + itoa(i) + "," + itoa(j) + 
                         "): expected " + std::to_string(expected) + 
                         ", got " + std::to_string(actual), 8);
            }
        }
    }
    
    printInfo("Product verification complete:", 4);
    printInfo("Total elements: 81", 8);
    printInfo("Matches: " + itoa(product_matches), 8);
    printInfo("Mismatches: " + itoa(product_mismatches), 8);
    
    if (product_mismatches == 0) {
        printInfo("✓ SUCCESS: Sparse matrix multiplication is correct!", 4);
    } else {
        printInfo("✗ FAILURE: Sparse matrix multiplication has errors!", 4);
        printInfo("FAILED TESTS", 0);
        return 1;
    }
    
    // Print sample of the product matrix
    printInfo("\nSample of product matrix (first 3x3 block):", 4);
    for (int i = 0; i < std::min(3, matrix_product_cpu->rows); i++) {
        std::string rowStr = "Row " + itoa(i) + ": ";
        for (int j = 0; j < std::min(3, matrix_product_cpu->cols); j++) {
            float val = getElementMatrixCSR(matrix_product_cpu, i, j);
            rowStr += std::to_string(val) + " ";
        }
        printInfo(rowStr, 8);
    }
    
    // Clean up temporary arrays
    for (int i = 0; i < 9; i++) {
        delete[] dense_A[i];
        delete[] expected_product[i];
    }
    delete[] dense_A;
    delete[] expected_product;
    
    // Clean up product matrix memory
    free(matrix_product_cpu->i);
    free(matrix_product_cpu->j);
    free(matrix_product_cpu->val);
    free(matrix_product_cpu);
    
    // Free GPU memory for product matrix
    freeMatrixCSRGPU(matrix_product_gpu);
    
    // Clean up transposed matrix memory
    free(matrix_transpose_cpu->i);
    free(matrix_transpose_cpu->j);
    free(matrix_transpose_cpu->val);
    free(matrix_transpose_cpu);
    
    // Free GPU memory for transposed matrix
    MatrixCSR* shallow_transpose_gpu = shallowCopyMatrixCSRGPUtoCPU(matrix_transpose_gpu);
    cudaFree(shallow_transpose_gpu->i);
    cudaFree(shallow_transpose_gpu->j);
    cudaFree(shallow_transpose_gpu->val);
    cudaFree(matrix_transpose_gpu);
    free(shallow_transpose_gpu);
    
    // Clean up memory
    printLines();
    printInfo("Cleaning up memory", 0);
    
    // Free CPU memory
    free(matrix_cpu->i);
    free(matrix_cpu->j);
    free(matrix_cpu->val);
    free(matrix_cpu);
    
    free(matrix_cpu_copy->i);
    free(matrix_cpu_copy->j);
    free(matrix_cpu_copy->val);
    free(matrix_cpu_copy);
    
    // Free CSC CPU memory
    free(matrix_csc_cpu->i);
    free(matrix_csc_cpu->j);
    free(matrix_csc_cpu->val);
    free(matrix_csc_cpu);
    
    // Free CSC GPU memory
    freeMatrixCSCGPU(matrix_csc_gpu);
    
    // Free GPU memory
    MatrixCSR* shallow_gpu = shallowCopyMatrixCSRGPUtoCPU(matrix_gpu);
    cudaFree(shallow_gpu->i);
    cudaFree(shallow_gpu->j);
    cudaFree(shallow_gpu->val);
    cudaFree(matrix_gpu);
    free(shallow_gpu);
    
    // Destroy cuSPARSE handle
    cusparseDestroy(cusparse_handle);
    
    printInfo("Test completed!", 0);
    printInfo("ALL TESTS PASSED SUCCESSFULLY!", 0);
    printLines();
    
    return 0;
} 