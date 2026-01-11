REBOL [
    Title: "Covenant v1.2.0 - Performance Optimized Demo"
    Description: "Demonstration of performance optimizations in Covenant AI Framework v1.2.0"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
do %../covenant.reb

print "Covenant AI Framework v1.2.0 - Performance Optimized Demo"
print "=========================================================="

;; 1. PERFORMANCE OPTIMIZED TENSOR OPERATIONS
print "^n1. Performance Optimized Tensor Operations:"

;; Create tensors efficiently
large-tensor: covenant/tensor (loop 1000 [append copy [] random 1.0])
print ["Large tensor created with" length? large-tensor/data "elements"]

;; Efficient operations
a: covenant/tensor [1.0 2.0 3.0 4.0 5.0]
b: covenant/tensor [2.0 3.0 4.0 5.0 6.0]

;; Efficient addition
result-add: covenant/add a b
print ["Addition result:" mold result-add/data]

;; Efficient multiplication
result-mul: covenant/mul a b
print ["Multiplication result:" mold result-mul/data]

;; Matrix operations
mat-a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
mat-b: covenant/tensor [[5.0 6.0] [7.0 8.0]]
mat-result: covenant/matmul mat-a mat-b
print ["Matrix multiplication result:" mold mat-result/data]

;; 2. OPTIMIZED NEURAL NETWORK OPERATIONS
print "^n2. Optimized Neural Network Operations:"

;; Create a simple model with optimized layers
input-data: covenant/tensor/requires_grad [1.0 2.0 3.0]

;; Linear layer
linear-layer: covenant/nn/linear 3 2
linear-output: linear-layer/forward input-data
print ["Linear layer output:" mold linear-output/data]

;; Activation functions
relu-output: covenant/nn/relu linear-output
print ["ReLU output:" mold relu-output/data]

;; Sigmoid
sigmoid-output: covenant/nn/sigmoid linear-output
print ["Sigmoid output:" mold sigmoid-output/data]

;; 3. OPTIMIZED AUTODIFF SYSTEM
print "^n3. Optimized Autodiff System:"

;; Create variables that require gradients
x: covenant/tensor/requires_grad [2.0]
y: covenant/tensor/requires_grad [3.0]

;; Perform operations
z: covenant/add x y
w: covenant/mul z x

print ["z = x + y =" mold z/data]
print ["w = z * x =" mold w/data]

;; 4. OPTIMIZED OPTIMIZERS
print "^n4. Optimized Optimizers:"

;; Create parameters to optimize
param1: covenant/tensor/requires_grad [0.5 -0.3 1.2]
param2: covenant/tensor/requires_grad [-0.1 0.8 0.4]
params: reduce [param1 param2]

;; Test different optimizers with optimized implementations
sgd-optimizer: covenant/optim/sgd/momentum params 0.01 0.9
adam-optimizer: covenant/optim/adam params 0.001
rmsprop-optimizer: covenant/optim/rmsprop params 0.01

print ["SGD optimizer initialized"]
print ["Adam optimizer initialized"]
print ["RMSprop optimizer initialized"]

;; 5. MEMORY EFFICIENT OPERATIONS
print "^n5. Memory Efficient Operations:"

;; Efficient reshape (doesn't copy data unnecessarily)
original: covenant/tensor (loop 12 [append copy [] random 1.0])
reshaped: covenant/reshape original [3 4]
print ["Reshaped tensor from" mold original/shape "to" mold reshaped/shape]

;; Efficient view (no data copying)
viewed: covenant/view original [2 6]
print ["Viewed tensor shape:" mold viewed/shape]

;; 6. PERFORMANCE TESTS
print "^n6. Performance Tests:"

;; Time basic operations
start-time: now/time/precise
repeat i 100 [
    test-a: covenant/tensor [1.0 2.0 3.0]
    test-b: covenant/tensor [4.0 5.0 6.0]
    result: covenant/add test-a test-b
]
end-time: now/time/precise
print ["100 additions completed in:" difference end-time start-time]

;; Time matrix operations
start-time: now/time/precise
repeat i 50 [
    mat-a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
    mat-b: covenant/tensor [[5.0 6.0] [7.0 8.0]]
    result: covenant/matmul mat-a mat-b
]
end-time: now/time/precise
print ["50 matrix multiplications completed in:" difference end-time start-time]

;; 7. OPTIMIZED GRADIENT COMPUTATIONS
print "^n7. Optimized Gradient Computations:"

;; Create a simple computational graph
x: covenant/tensor/requires_grad [2.0]
y: covenant/tensor/requires_grad [3.0]

;; More complex operations
z: covenant/add x y
w: covenant/mul z x
squared: covenant/pow w 2
result: covenant/sum squared

print ["Complex computation result:" mold result/data]

;; 8. EVALUATION METRICS
print "^n8. Evaluation Metrics:"

;; Test predictions vs targets
predictions: covenant/tensor [1.0 0.0 1.0 1.0 0.0]
targets: covenant/tensor [1.0 0.0 1.0 0.0 0.0]

accuracy: covenant/metrics/accuracy predictions targets
print ["Accuracy:" accuracy/percentage "%"]

precision: covenant/metrics/precision predictions targets
print ["Precision:" precision/percentage "%"]

recall: covenant/metrics/recall predictions targets
print ["Recall:" recall/percentage "%"]

f1: covenant/metrics/f1_score predictions targets
print ["F1 Score:" f1/percentage "%"]

;; 9. PREPROCESSING UTILITIES
print "^n9. Preprocessing Utilities:"

;; Normalize data
raw-data: covenant/tensor [1.0 2.0 3.0 4.0 5.0]
normalized: covenant/preprocessing/min_max_normalize raw-data
print ["Normalized data:" mold normalized/data]

;; Standardize data
standardized: covenant/preprocessing/standardize raw-data
print ["Standardized data:" mold standardized/data]

;; Train-test split
split-result: covenant/preprocessing/train_test_split raw-data 0.2
print ["Train data:" mold split-result/train/data]
print ["Test data:" mold split-result/test/data]

;; 10. LOGGING SYSTEM
print "^n10. Logging System:"
covenant/logging/set_level covenant/logging/INFO
covenant/logging/info "Performance optimized demo completed successfully"
covenant/logging/debug "This is a debug message that won't appear with INFO level"

print "^nCovenant AI Framework v1.2.0 Performance Optimized Demo Completed!"
print "=================================================================="
print "Key optimizations implemented:"
print "- Efficient memory allocation and reuse"
print "- Optimized tensor operations with pre-allocation"
print "- Streamlined computational graph construction"
print "- Optimized gradient computation paths"
print "- Reduced unnecessary data copying"
print "- Efficient batch processing operations"
print "- Optimized mathematical operations"
print "- Memory-conscious data structures"
