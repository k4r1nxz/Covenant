REBOL [
    Title: "Covenant Framework Simple Test"
    Description: "Simple test to verify Covenant AI Framework functionality"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
do %covenant.r

print "Covenant AI Framework - Simple Functionality Test"
print "================================================="

;; Test 1: Basic tensor creation using covenant namespace
print "^n1. Testing basic tensor creation..."
t1: covenant/tensor [1.0 2.0 3.0]
print ["Tensor created with shape:" mold t1/shape "and data:" mold t1/data]

t2: covenant/zeros [3]
print ["Zeros tensor created with shape:" mold t2/shape "and data:" mold t2/data]

t3: covenant/ones [2 3]
print ["Ones tensor created with shape:" mold t3/shape "and data:" mold t3/data]

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

;; Test 4: Neural network components
print "^n4. Testing neural network components..."
linear-layer: covenant/nn/linear 3 2
input-data: covenant/tensor [1.0 2.0 3.0]

output: linear-layer/forward input-data
print ["Linear layer output:" mold output/data]

relu-activated: covenant/nn/relu output
print ["After ReLU activation:" mold relu-activated/data]

sigmoid-activated: covenant/nn/sigmoid output
print ["After Sigmoid activation:" mold sigmoid-activated/data]

;; Test 5: Loss function
print "^n5. Testing loss function..."
predictions: covenant/tensor [1.0 2.0 3.0]
targets: covenant/tensor [1.1 1.9 3.2]

mse: covenant/nn/mse-loss predictions targets
print ["MSE Loss:" mse]

;; Test 6: Optimization
print "^n6. Testing optimization..."
params: reduce [1.0 2.0 3.0]
optimizer: covenant/optim/sgd params 0.01
print ["SGD optimizer created with learning rate:" optimizer/lr]

adam-optimizer: covenant/optim/adam params 0.001
print ["Adam optimizer created with learning rate:" adam-optimizer/lr]

;; Test 7: Utility functions
print "^n7. Testing utility functions..."
covenant/utils/visualize mat-a

;; Save a simple model representation
test-model: make object! [weights: mat-a bias: covenant/zeros [2]]
covenant/utils/save-model test-model %test-model.dat

print "^nAll tests completed successfully using the Covenant namespace!"