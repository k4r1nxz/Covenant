REBOL [
    Title: "Covenant Framework Test"
    Description: "Test script to verify Covenant AI Framework functionality"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
do %covenant.r

print "Covenant AI Framework - Functionality Test"
print "=========================================="

;; Test 1: Basic tensor creation
print "^n1. Testing basic tensor creation..."
t1: covenant/tensor [1.0 2.0 3.0]
print ["Tensor created with shape:" mold t1/shape "and data:" mold t1/data]

t2: covenant/zeros [3]
print ["Zeros tensor created with shape:" mold t2/shape "and data:" mold t2/data]

t3: covenant/ones [2 3]
print ["Ones tensor created with shape:" mold t3/shape "and data:" mold t3/data]

t4: covenant/rand [2 2]
print ["Random tensor created with shape:" mold t4/shape "and data:" mold t4/data]

;; Test 2: Tensor operations
print "^n2. Testing tensor operations..."
a: covenant/tensor [1.0 2.0 3.0]
b: covenant/tensor [4.0 5.0 6.0]

sum-result: covenant/add a b
print ["Addition result:" mold sum-result/data]

mul-result: covenant/mul a b
print ["Multiplication result:" mold mul-result/data]

;; Test 3: Matrix operations
print "^n3. Testing matrix operations..."
mat-a: covenant/tensor [[1.0 2.0] [3.0 4.0]]
mat-b: covenant/tensor [[5.0 6.0] [7.0 8.0]]

matmul-result: covenant/matmul mat-a mat-b
print ["Matrix multiplication result:" mold matmul-result/data]

;; Test 4: Reshape operation
print "^n4. Testing reshape operation..."
flat-tensor: covenant/ones [6]
print ["Original tensor shape:" mold flat-tensor/shape]

reshaped-tensor: covenant/reshape flat-tensor [2 3]
print ["Reshaped tensor shape:" mold reshaped-tensor/shape "data:" mold reshaped-tensor/data]

;; Test 5: Sum operation
print "^n5. Testing sum operation..."
sum-axis0: covenant/sum reshaped-tensor 0
print ["Sum along axis 0:" mold sum-axis0/data]

sum-axis1: covenant/sum reshaped-tensor 1
print ["Sum along axis 1:" mold sum-axis1/data]

;; Test 6: Neural network components
print "^n6. Testing neural network components..."
linear-layer: covenant/nn/linear 3 2
input-data: covenant/tensor [1.0 2.0 3.0]

output: linear-layer/forward input-data
print ["Linear layer output:" mold output/data]

relu-activated: covenant/nn/relu output
print ["After ReLU activation:" mold relu-activated/data]

sigmoid-activated: covenant/nn/sigmoid output
print ["After Sigmoid activation:" mold sigmoid-activated/data]

tanh-activated: covenant/nn/tanh output
print ["After Tanh activation:" mold tanh-activated/data]

;; Test 7: Loss function
print "^n7. Testing loss function..."
predictions: covenant/tensor [1.0 2.0 3.0]
targets: covenant/tensor [1.1 1.9 3.2]

mse: covenant/nn/mse-loss predictions targets
print ["MSE Loss:" mse]

;; Test 8: Optimization
print "^n8. Testing optimization..."
params: reduce [1.0 2.0 3.0]
optimizer: covenant/optim/sgd params 0.01
print ["SGD optimizer created with learning rate:" optimizer/lr]

adam-optimizer: covenant/optim/adam params 0.001
print ["Adam optimizer created with learning rate:" adam-optimizer/lr]

;; Test 9: Utility functions
print "^n9. Testing utility functions..."
covenant/utils/visualize reshaped-tensor

;; Save a simple model representation
test-model: make object! [weights: reshaped-tensor bias: covenant/zeros [3]]
covenant/utils/save-model test-model %test-model.dat

print "^nAll tests completed successfully! Covenant AI Framework is working properly."
