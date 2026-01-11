REBOL [
    Title: "Covenant Optimization Module"
    Description: "Optimization algorithms for Covenant AI Framework"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
    Rights: "BSD 2-Clause License"
]

;; Optimization algorithms
optim: context [
    ;; Stochastic Gradient Descent - optimized version
    sgd: func [
        "Stochastic Gradient Descent optimizer"
        param-list [block!] "Parameters to optimize"
        lr [number!] "Learning rate"
        /momentum "Use momentum"
        mom [number!] "Momentum factor (default 0.0)"
        /nesterov "Use Nesterov momentum"
    ] [
        params-local: copy param-list
        lr-local: lr
        momentum-local: either momentum [mom] [0.0]
        nesterov-local: either nesterov [true] [false]
        
        ; Initialize velocity if momentum is used
        velocity: make block! length? param-list
        loop length? param-list [append velocity 0.0]

        make object! [
            type: 'sgd
            params: params-local
            lr: lr-local
            momentum: momentum-local
            nesterov: nesterov-local
            velocity: velocity

            step: func [gradients] [
                len: length? params-local
                repeat idx len [
                    g: gradients/:idx
                    
                    ; Update velocity
                    vel: velocity/:idx
                    new-vel: (momentum-local * vel) - (lr-local * g)
                    velocity/:idx: new-vel
                    
                    ; If using Nesterov momentum, add additional term
                    if nesterov-local [
                        new-vel: new-vel - (momentum-local * lr-local * g)
                    ]
                    
                    ; Update parameters
                    params-local/:idx: params-local/:idx + new-vel
                ]
            ]
            
            zero-grad: func [] [
                ; Reset velocity
                len: length? velocity
                repeat idx len [
                    velocity/:idx: 0.0
                ]
            ]
        ]
    ]

    ;; Adam optimizer (improved implementation) - optimized version
    adam: func [
        "Adam optimizer"
        param-list [block!] "Parameters to optimize"
        lr [number!] "Learning rate"
        /beta1 "Beta1 parameter (default 0.9)"
        b1 [number!] "Beta1 value"
        /beta2 "Beta2 parameter (default 0.999)"
        b2 [number!] "Beta2 value"
        /epsilon "Epsilon parameter (default 1e-8)"
        eps [number!] "Epsilon value"
    ] [
        params-local: copy param-list
        m-local: make block! length? param-list  ; First moment
        v-local: make block! length? param-list  ; Second moment
        t-local: 0  ; Timestep

        ; Initialize moments to zero
        len: length? param-list
        loop len [
            append m-local 0.0
            append v-local 0.0
        ]

        lr-local: lr
        beta1-local: either beta1 [b1] [0.9]
        beta2-local: either beta2 [b2] [0.999]
        epsilon-local: either epsilon [eps] [1e-8]

        make object! [
            type: 'adam
            params: params-local
            lr: lr-local
            beta1: beta1-local
            beta2: beta2-local
            epsilon: epsilon-local
            m: m-local
            v: v-local
            t: t-local

            step: func [gradients] [
                t-local: t-local + 1
                len: length? params-local
                repeat idx len [
                    g: gradients/:idx

                    ; Update biased first moment estimate
                    m-val: m-local/:idx
                    new-m: (beta1-local * m-val) + ((1 - beta1-local) * g)
                    m-local/:idx: new-m

                    ; Update biased second raw moment estimate
                    v-val: v-local/:idx
                    g-sq: g * g
                    new-v: (beta2-local * v-val) + ((1 - beta2-local) * g-sq)
                    v-local/:idx: new-v

                    ; Compute bias-corrected first moment estimate
                    m-hat: new-m / (1 - (power beta1-local t-local))

                    ; Compute bias-corrected second raw moment estimate
                    v-hat: new-v / (1 - (power beta2-local t-local))

                    ; Update parameters
                    sqrt-v-eps: core/square-root (v-hat + epsilon-local)
                    params-local/:idx: params-local/:idx - (lr-local * m-hat / sqrt-v-eps)
                ]
            ]
            
            zero-grad: func [] [
                ; Reset moments
                len: length? m-local
                repeat idx len [
                    m-local/:idx: 0.0
                    v-local/:idx: 0.0
                ]
                t-local: 0
            ]
        ]
    ]

    ;; RMSprop optimizer - optimized version
    rmsprop: func [
        "RMSprop optimizer"
        param-list [block!] "Parameters to optimize"
        lr [number!] "Learning rate"
        /alpha "Smoothing constant (default 0.999)"
        a [number!] "Alpha value"
        /epsilon "Epsilon parameter (default 1e-8)"
        eps [number!] "Epsilon value"
    ] [
        params-local: copy param-list
        v-local: make block! length? param-list  ; Running average of squared gradients
        t-local: 0  ; Timestep

        ; Initialize to zero
        len: length? param-list
        loop len [append v-local 0.0]

        lr-local: lr
        alpha-local: either alpha [a] [0.999]
        epsilon-local: either epsilon [eps] [1e-8]

        make object! [
            type: 'rmsprop
            params: params-local
            lr: lr-local
            alpha: alpha-local
            epsilon: epsilon-local
            v: v-local
            t: t-local

            step: func [gradients] [
                t-local: t-local + 1
                len: length? params-local
                repeat idx len [
                    g: gradients/:idx

                    ; Update running average of squared gradients
                    v-val: v-local/:idx
                    g-sq: g * g
                    new-v: (alpha-local * v-val) + ((1 - alpha-local) * g-sq)
                    v-local/:idx: new-v

                    ; Update parameters
                    sqrt-v-eps: core/square-root (new-v + epsilon-local)
                    params-local/:idx: params-local/:idx - (lr-local * g / sqrt-v-eps)
                ]
            ]
            
            zero-grad: func [] [
                ; Reset running average
                len: length? v-local
                repeat idx len [
                    v-local/:idx: 0.0
                ]
            ]
        ]
    ]

    ;; Adagrad optimizer - optimized version
    adagrad: func [
        "Adagrad optimizer"
        param-list [block!] "Parameters to optimize"
        lr [number!] "Learning rate"
        /epsilon "Epsilon parameter (default 1e-8)"
        eps [number!] "Epsilon value"
    ] [
        params-local: copy param-list
        v-local: make block! length? param-list  ; Sum of squared gradients
        t-local: 0  ; Timestep

        ; Initialize to zero
        len: length? param-list
        loop len [append v-local 0.0]

        lr-local: lr
        epsilon-local: either epsilon [eps] [1e-8]

        make object! [
            type: 'adagrad
            params: params-local
            lr: lr-local
            epsilon: epsilon-local
            v: v-local
            t: t-local

            step: func [gradients] [
                t-local: t-local + 1
                len: length? params-local
                repeat idx len [
                    g: gradients/:idx

                    ; Update sum of squared gradients
                    v-val: v-local/:idx
                    g-sq: g * g
                    new-v: v-val + g-sq
                    v-local/:idx: new-v

                    ; Update parameters
                    sqrt-v-eps: core/square-root (new-v + epsilon-local)
                    params-local/:idx: params-local/:idx - (lr-local * g / sqrt-v-eps)
                ]
            ]
            
            zero-grad: func [] [
                ; Reset sum of squared gradients
                len: length? v-local
                repeat idx len [
                    v-local/:idx: 0.0
                ]
            ]
        ]
    ]

    ;; Adadelta optimizer - optimized version
    adadelta: func [
        "Adadelta optimizer"
        param-list [block!] "Parameters to optimize"
        /rho "Rho parameter (default 0.95)"
        r [number!] "Rho value"
        /epsilon "Epsilon parameter (default 1e-6)"
        eps [number!] "Epsilon value"
    ] [
        params-local: copy param-list
        v-local: make block! length? param-list  ; Running average of squared gradients
        delta-local: make block! length? param-list  ; Running average of squared parameter updates
        t-local: 0  ; Timestep

        ; Initialize to zero
        len: length? param-list
        loop len [
            append v-local 0.0
            append delta-local 0.0
        ]

        rho-local: either rho [r] [0.95]
        epsilon-local: either epsilon [eps] [1e-6]

        make object! [
            type: 'adadelta
            params: params-local
            rho: rho-local
            epsilon: epsilon-local
            v: v-local
            delta: delta-local
            t: t-local

            step: func [gradients] [
                t-local: t-local + 1
                len: length? params-local
                repeat idx len [
                    g: gradients/:idx

                    ; Update running average of squared gradients
                    v-val: v-local/:idx
                    g-sq: g * g
                    new-v: (rho-local * v-val) + ((1 - rho-local) * g-sq)
                    v-local/:idx: new-v

                    ; Calculate parameter update
                    sqrt-delta-eps: core/square-root (delta-local/:idx + epsilon-local)
                    sqrt-v-eps: core/square-root (new-v + epsilon-local)
                    param-update: - (sqrt-delta-eps / sqrt-v-eps) * g

                    ; Update running average of squared parameter updates
                    delta-val: delta-local/:idx
                    update-sq: param-update * param-update
                    new-delta: (rho-local * delta-val) + ((1 - rho-local) * update-sq)
                    delta-local/:idx: new-delta

                    ; Update parameters
                    params-local/:idx: params-local/:idx + param-update
                ]
            ]
            
            zero-grad: func [] [
                ; Reset running averages
                len: length? v-local
                repeat idx len [
                    v-local/:idx: 0.0
                    delta-local/:idx: 0.0
                ]
            ]
        ]
    ]

    ;; Learning rate scheduler
    lr-scheduler: context [
        ;; Step learning rate scheduler - optimized version
        step-lr: func [
            "Step learning rate scheduler"
            optimizer [object!] "Optimizer to schedule"
            step-size [integer!] "Period of learning rate decay"
            gamma [number!] "Multiplicative factor of learning rate decay (default 0.1)"
        ] [
            opt-ref: optimizer
            step-size-local: step-size
            gamma-local: either gamma [gamma] [0.1]
            last-epoch: 0

            make object! [
                optimizer: opt-ref
                step-size: step-size-local
                gamma: gamma-local
                last-epoch: last-epoch

                step: func [epoch] [
                    if (epoch // step-size-local) = 0 [
                        if epoch > last-epoch [
                            opt-ref/lr: opt-ref/lr * gamma-local
                            last-epoch: epoch
                        ]
                    ]
                ]
            ]
        ]

        ;; Exponential learning rate scheduler - optimized version
        exp-lr: func [
            "Exponential learning rate scheduler"
            optimizer [object!] "Optimizer to schedule"
            gamma [number!] "Multiplicative factor of learning rate decay"
        ] [
            opt-ref: optimizer
            gamma-local: gamma
            last-epoch: 0

            make object! [
                optimizer: opt-ref
                gamma: gamma-local
                last-epoch: last-epoch

                step: func [epoch] [
                    if epoch > last-epoch [
                        opt-ref/lr: opt-ref/lr * gamma-local
                        last-epoch: epoch
                    ]
                ]
            ]
        ]
    ]

]
