REBOL [
    Title: "Covenant AI Framework"
    Description: "A REBOL-based AI framework for machine learning and neural networks"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
    Rights: "BSD 2-Clause License"
]

;; Import core modules
do %core/core.reb
do %core/autograd.reb
do %core/variable.reb
do %core/backward.reb
do %core/operation-gradients.reb
do %core/central-graph.reb
do %nn/nn.reb
do %optim/optim.reb
do %utils/utils.reb

;; Define function references first
tensor-func: get in core 'tensor
zeros-func: get in core 'zeros
ones-func: get in core 'ones
rand-func: get in core 'rand
reshape-func: get in core 'reshape
sum-func: get in core 'sum
add-func: get in core 'add
mul-func: get in core 'mul
matmul-func: get in core 'matmul

;; Neural network functions
linear-func: get in nn 'linear-grad
relu-func: get in nn 'relu
sigmoid-func: get in nn 'sigmoid
tanh-func: get in nn 'tanh
softmax-func: get in nn 'softmax
mse-loss-func: get in nn 'mse-loss
conv1d-func: get in nn 'conv1d
maxpool1d-func: get in nn 'maxpool1d
dropout-func: get in nn 'dropout
batchnorm1d-func: get in nn 'batchnorm1d

;; Optimization functions
sgd-func: get in optim 'sgd
adam-func: get in optim 'adam

;; Utility functions
load-data-func: get in utils 'load-data
save-model-func: get in utils 'save-model
visualize-func: get in utils 'visualize
load-model-func: get in utils 'load-model
evaluate-func: get in utils 'evaluate

;; Define the covenant namespace after all modules are loaded
covenant: make object! [
    ;; Core tensor operations
    tensor: func [
        "Create a tensor with optional gradient tracking"
        data [block! number!] "Input data"
        /requires_grad "Enable gradient tracking"
    ] [
        either requires_grad [
            core/tensor-grad/requires-grad data
        ] [
            tensor-func data
        ]
    ]
    zeros: :zeros-func
    ones: :ones-func
    rand: :rand-func
    reshape: :reshape-func
    sum: :sum-func
    add: :add-func
    mul: :mul-func
    matmul: :matmul-func
    concat: get in core 'concat
    stack: get in core 'stack
    mean: get in core 'mean
    flatten: get in core 'flatten
    view: get in core 'view

    ;; Neural network components
    nn: make object! [
        linear: :linear-func
        relu: :relu-func
        sigmoid: :sigmoid-func
        tanh: :tanh-func
        softmax: :softmax-func
        mse-loss: :mse-loss-func
        conv1d: :conv1d-func
        maxpool1d: :maxpool1d-func
        dropout: :dropout-func
        batchnorm1d: :batchnorm1d-func
    ]

    ;; Central graph manager access
    graph: :graph-manager

    ;; Optimization utilities
    optim: make object! [
        sgd: :sgd-func
        adam: :adam-func
    ]

    ;; Utility functions
    utils: make object! [
        load-data: :load-data-func
        save-model: :save-model-func
        visualize: :visualize-func
        load-model: :load-model-func
        evaluate: :evaluate-func
    ]
]

print "Covenant AI Framework loaded successfully"