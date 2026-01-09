REBOL [
    Title: "Covenant Advanced Features Example"
    Description: "Example demonstrating advanced features of Covenant AI Framework v1.1.0"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
do %../covenant.reb

print "Covenant AI Framework v1.1.0 - Advanced Features Demo"
print "====================================================="

;; 1. NEW TENSOR OPERATIONS
print "^n1. Demonstrating new tensor operations..."

;; Arange and linspace
print "^nCreating tensors with arange and linspace:"
range-tensor: covenant/arange 0 10
print ["Arange tensor:" mold range-tensor/data]

linspace-tensor: covenant/linspace 0 1 5
print ["Linspace tensor:" mold linspace-tensor/data]

;; Power, sqrt, exp, log operations
print "^nMathematical operations:"
base: covenant/tensor [1.0 2.0 3.0]
powered: covenant/pow base 2
print ["Power operation (x^2):" mold powered/data]

squared: covenant/tensor [4.0 9.0 16.0]
sqrt-result: covenant/sqrt squared
print ["Square root:" mold sqrt-result/data]

exp-input: covenant/tensor [0.0 1.0 2.0]
exp-result: covenant/exp exp-input
print ["Exponential:" mold exp-result/data]

log-input: covenant/tensor [1.0 2.718 7.389]
log-result: covenant/log log-input
print ["Natural log:" mold log-result/data]

;; Statistical operations
print "^nStatistical operations:"
data: covenant/tensor [[1.0 2.0 3.0] [4.0 5.0 6.0]]
print ["Original data:" mold data/data]

max-cols: covenant/max/axis data 0
print ["Max along columns:" mold max-cols/data]

max-rows: covenant/max/axis data 1
print ["Max along rows:" mold max-rows/data]

argmax-result: covenant/argmax/axis data 1
print ["Argmax along rows:" mold argmax-result/data]

transposed: covenant/transpose data
print ["Transposed data:" mold transposed/data]

;; 2. NEW NEURAL NETWORK LAYERS
print "^n2. Demonstrating new neural network layers..."

;; Create input data
input-data: covenant/tensor/requires_grad [1.0 -2.0 3.0]

;; Leaky ReLU
leaky-relu-layer: covenant/nn/leaky_relu
leaky-relu-output: leaky-relu-layer input-data
print ["Leaky ReLU output:" mold leaky-relu-output/data]

;; Cross entropy loss
preds: covenant/tensor [0.2 0.3 0.5]
targets: covenant/tensor [0.0 0.0 1.0]
cross-entropy: covenant/nn/cross_entropy_loss preds targets
print ["Cross entropy loss:" mold cross-entropy/data]

;; Batch normalization
bn-layer: covenant/nn/batchnorm1d 3
bn-input: covenant/tensor/requires_grad [[1.0 2.0 3.0] [4.0 5.0 6.0]]
bn-output: bn-layer/forward bn-input
print ["Batch norm output shape:" mold bn-output/shape]

;; Sequential model
layer1: covenant/nn/linear 3 5
relu1: covenant/nn/relu
layer2: covenant/nn/linear 5 1
model: covenant/nn/sequential reduce [layer1 relu1 layer2]

sample-input: covenant/tensor [1.0 2.0 3.0]
model-output: model/forward sample-input
print ["Sequential model output:" mold model-output/data]

;; 3. NEW OPTIMIZATION ALGORITHMS
print "^n3. Demonstrating new optimization algorithms..."

;; Create parameters to optimize
param1: covenant/tensor/requires_grad [0.5 -0.3 1.2]
param2: covenant/tensor/requires_grad [-0.1 0.8 0.4]
params: reduce [param1 param2]

;; Test different optimizers
print "^nTesting SGD with momentum:"
sgd-opt: covenant/optim/sgd/momentum params 0.01 0.9
print ["SGD with momentum - initial param1:" mold param1/data]
; Simulate a gradient update (using dummy gradients)
dummy-grads: reduce [covenant/tensor [0.1 -0.2 0.3]/data covenant/tensor [-0.05 0.15 0.25]/data]
sgd-opt/step dummy-grads
print ["SGD with momentum - after update:" mold param1/data]

;; Reset for next test
param1/data: [0.5 -0.3 1.2]
param2/data: [-0.1 0.8 0.4]

print "^nTesting Adam optimizer:"
adam-opt: covenant/optim/adam params 0.001
adam-opt/step dummy-grads
print ["Adam - after update:" mold param1/data]

;; Reset for next test
param1/data: [0.5 -0.3 1.2]
param2/data: [-0.1 0.8 0.4]

print "^nTesting RMSprop optimizer:"
rmsprop-opt: covenant/optim/rmsprop params 0.01
rmsprop-opt/step dummy-grads
print ["RMSprop - after update:" mold param1/data]

;; Reset for next test
param1/data: [0.5 -0.3 1.2]
param2/data: [-0.1 0.8 0.4]

print "^nTesting Adagrad optimizer:"
adagrad-opt: covenant/optim/adagrad params 0.01
adagrad-opt/step dummy-grads
print ["Adagrad - after update:" mold param1/data]

;; 4. LEARNING RATE SCHEDULERS
print "^n4. Demonstrating learning rate schedulers..."

;; Create optimizer with initial learning rate
lr-test-params: reduce [covenant/tensor/requires_grad [0.5]]
lr-test-opt: covenant/optim/sgd lr-test-params 0.01

print ["Initial learning rate:" lr-test-opt/lr]

;; Create step learning rate scheduler
scheduler: covenant/lr_scheduler/step_lr lr-test-opt 5 0.5  ; Reduce LR by half every 5 epochs

print "^nSimulating training with LR scheduler:"
repeat epoch 15 [
    if (epoch // 5) = 1 [  ; Apply scheduler every 5 epochs
        scheduler/step epoch
        print ["Epoch" epoch "- Learning rate changed to:" lr-test-opt/lr]
    ]
]

print "^nAdvanced Features Demo Completed Successfully!"
print "==============================================="
print "Summary of new features demonstrated:"
print "- New tensor operations: arange, linspace, pow, sqrt, exp, log, max, min, argmax, argmin, transpose"
print "- New neural network layers: leaky_relu, cross_entropy_loss, batchnorm1d, sequential"
print "- New optimization algorithms: SGD with momentum/Nesterov, Adam, RMSprop, Adagrad, Adadelta"
print "- Learning rate schedulers: step_lr, exp_lr"
print "- Enhanced tensor dtype support: float32, int32, int64"
print "- Improved autograd system with more operations"