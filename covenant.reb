REBOL [
    Title: "Covenant AI Framework"
    Description: "A REBOL-based AI framework for machine learning and neural networks"
    Version: 1.1.0
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

;; Additional tensor operations
arange-func: get in core 'arange
linspace-func: get in core 'linspace
pow-func: get in core 'pow
sqrt-func: get in core 'sqrt
exp-func: get in core 'exp
log-func: get in core 'log
max-func: get in core 'max
min-func: get in core 'min
argmax-func: get in core 'argmax
argmin-func: get in core 'argmin
transpose-func: get in core 'transpose

;; Neural network functions
linear-func: get in nn 'linear-grad
relu-func: get in nn 'relu
leaky-relu-func: get in nn 'leaky-relu
sigmoid-func: get in nn 'sigmoid
tanh-func: get in nn 'tanh
softmax-func: get in nn 'softmax
mse-loss-func: get in nn 'mse-loss
cross-entropy-loss-func: get in nn 'cross-entropy-loss
conv1d-func: get in nn 'conv1d
dropout-func: get in nn 'dropout
batchnorm1d-func: get in nn 'batchnorm1d
sequential-func: get in nn 'sequential

;; Optimization functions
sgd-func: get in optim 'sgd
adam-func: get in optim 'adam
rmsprop-func: get in optim 'rmsprop
adagrad-func: get in optim 'adagrad
adadelta-func: get in optim 'adadelta

;; Learning rate schedulers
step-lr-func: get in optim/lr-scheduler 'step-lr
exp-lr-func: get in optim/lr-scheduler 'exp-lr

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
        /dtype "Specify data type"
        data-type [word!] "Data type (float32, int32, int64)"
    ] [
        either requires_grad [
            core/tensor-grad/requires-grad data
        ] [
            either dtype [
                core/tensor/dtype data data-type
            ] [
                tensor-func data
            ]
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

    ;; Additional tensor operations
    arange: :arange-func
    linspace: :linspace-func
    pow: :pow-func
    sqrt: :sqrt-func
    exp: :exp-func
    log: :log-func
    max: :max-func
    min: :min-func
    argmax: :argmax-func
    argmin: :argmin-func
    transpose: :transpose-func

    ;; Neural network components
    nn: make object! [
        linear: :linear-func
        relu: :relu-func
        leaky_relu: :leaky-relu-func
        sigmoid: :sigmoid-func
        tanh: :tanh-func
        softmax: :softmax-func
        mse_loss: :mse-loss-func
        cross_entropy_loss: :cross-entropy-loss-func
        conv1d: :conv1d-func
        dropout: :dropout-func
        batchnorm1d: :batchnorm1d-func
        sequential: :sequential-func
    ]

    ;; Central graph manager access
    graph: :graph-manager

    ;; Optimization utilities
    optim: make object! [
        sgd: :sgd-func
        adam: :adam-func
        rmsprop: :rmsprop-func
        adagrad: :adagrad-func
        adadelta: :adadelta-func
    ]

    ;; Learning rate schedulers
    lr_scheduler: make object! [
        step_lr: :step-lr-func
        exp_lr: :exp-lr-func
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

print "Covenant AI Framework v1.1.0 loaded successfully"
