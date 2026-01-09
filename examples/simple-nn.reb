REBOL [
    Title: "Covenant Example - Simple Neural Network"
    Description: "Example demonstrating Covenant AI Framework capabilities"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
do %../covenant.reb

print "Covenant AI Framework Example"
print "=============================="

;; Create some sample data
print "Creating sample data..."
input-data: covenant/tensor [1.0 2.0 3.0]
target-data: covenant/tensor [0.5 1.0]

print ["Input data:" mold input-data/data]
print ["Target data:" mold target-data/data]

;; Create a simple neural network
print "^/Creating a simple neural network..."
layer1: covenant/nn/linear 3 2  ; Input size 3, output size 2
layer2: covenant/nn/linear 2 1  ; Input size 2, output size 1

;; Forward pass through the network
print "^nPerforming forward pass..."
hidden: layer1/forward input-data
print ["Hidden layer output:" mold hidden/data]

activated: covenant/nn/relu hidden
print ["After ReLU activation:" mold activated/data]

output: layer2/forward activated
print ["Final output:" mold output/data]

;; Calculate loss
print "^nCalculating loss..."
loss: covenant/nn/mse-loss output target-data
print ["MSE Loss:" loss]

;; Create optimizer and perform a training step
print "^nCreating optimizer and performing training step..."
; Note: In this simplified example, we're not implementing full backpropagation
; but showing how the optimizer would be used in a real implementation
params: reduce [layer1/weights layer1/bias layer2/weights layer2/bias]
optimizer: covenant/optim/sgd params 0.01

print "Covenant framework example completed successfully!"
print "^nFramework features demonstrated:"
print "- Tensor operations (creation, addition, multiplication)"
print "- Neural network layers (linear, activation functions)"
print "- Loss functions (MSE)"
print "- Optimization algorithms (SGD)"
