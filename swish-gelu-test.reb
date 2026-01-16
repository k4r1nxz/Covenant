REBOL [
    Title: "Swish and GELU Activation Functions Test"
    Description: "Test script for new Swish and GELU activation functions in Covenant"
    Version: 1.3.0
    Author: "AI Assistant"
]

;; Load the Covenant framework
do %covenant.reb

print "Testing Swish and GELU Activation Functions"
print "==========================================="

;; Create a test tensor with various values
test-input: covenant/tensor [-2.0 -1.0 0.0 1.0 2.0]

print ["^nInput tensor: " mold test-input/data]

;; Test Swish activation
print "^n1. Testing Swish activation..."
swish-output: covenant/nn/swish test-input
print ["Swish output: " mold swish-output/data]

;; Test Swish activation with custom beta parameter
print "^n2. Testing Swish activation with beta=2.0..."
swish-output-beta: covenant/nn/swish/beta test-input 2.0
print ["Swish output (beta=2.0): " mold swish-output-beta/data]

;; Test GELU activation
print "^n3. Testing GELU activation..."
gelu-output: covenant/nn/gelu test-input
print ["GELU output: " mold gelu-output/data]

;; Compare with other activations
print "^n4. Comparison with other activation functions..."

;; Sigmoid
sigmoid-output: covenant/nn/sigmoid test-input
print ["Sigmoid output: " mold sigmoid-output/data]

;; Tanh
tanh-output: covenant/nn/tanh test-input
print ["Tanh output: " mold tanh-output/data]

;; ReLU
relu-output: covenant/nn/relu test-input
print ["ReLU output: " mold relu-output/data]

print "^nAll tests completed successfully!"
print "Swish and GELU functions are now available in the Covenant framework."
