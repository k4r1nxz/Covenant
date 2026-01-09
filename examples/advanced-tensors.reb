REBOL [
    Title: "Covenant Example - Advanced Tensor Operations"
    Description: "Example demonstrating advanced tensor operations in Covenant AI Framework"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
do %../covenant.rebeb

print "Covenant AI Framework - Advanced Tensor Operations"
print "=================================================="

;; Create 2D tensors (matrices)
print "Creating 2D tensors..."
matrix-a: covenant/tensor [[1.0 2.0 3.0] [4.0 5.0 6.0]]
matrix-b: covenant/tensor [[7.0 8.0] [9.0 10.0] [11.0 12.0]]

print ["Matrix A shape:" mold matrix-a/shape]
print ["Matrix A data:" mold matrix-a/data]

print ["Matrix B shape:" mold matrix-b/shape]
print ["Matrix B data:" mold matrix-b/data]

;; Matrix multiplication
print "^nPerforming matrix multiplication (A * B)..."
result: covenant/core/matmul matrix-a matrix-b
print ["Result shape:" mold result/shape]
print ["Result data:" mold result/data]

;; Tensor operations
print "^nCreating a tensor and reshaping it..."
tensor-1d: covenant/ones [6]
print ["Original 1D tensor shape:" mold tensor-1d/shape]
print ["Original 1D tensor data:" mold tensor-1d/data]

reshaped: covenant/core/reshape tensor-1d [2 3]
print ["Reshaped tensor shape:" mold reshaped/shape]
print ["Reshaped tensor data:" mold reshaped/data]

;; Sum operations
print "^nPerforming sum operations on the reshaped tensor..."
sum-axis-0: covenant/core/sum reshaped 0  ; Sum along rows
print ["Sum along axis 0 (rows):" mold sum-axis-0/data]

sum-axis-1: covenant/core/sum reshaped 1  ; Sum along columns
print ["Sum along axis 1 (columns):" mold sum-axis-1/data]

;; Element-wise operations
print "^nPerforming element-wise operations..."
tensor-x: covenant/tensor [1.0 2.0 3.0 4.0]
tensor-y: covenant/tensor [2.0 3.0 4.0 5.0]

added: covenant/core/add tensor-x tensor-y
print ["X + Y =" mold added/data]

multiplied: covenant/core/mul tensor-x tensor-y
print ["X * Y =" mold multiplied/data]

;; Neural network example with 2D tensors
print "^nCreating a simple neural network with 2D inputs..."
input-layer: covenant/nn/linear 4 2
input-data: covenant/tensor [1.0 2.0 3.0 4.0]

output: input-layer/forward input-data
activated: covenant/nn/sigmoid output

print ["Input data:" mold input-data/data]
print ["Linear output:" mold output/data]
print ["After sigmoid activation:" mold activated/data]

print "^nCovenant framework advanced operations example completed!"
