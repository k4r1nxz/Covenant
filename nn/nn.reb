REBOL [
    Title: "Covenant Neural Network Module"
    Description: "Neural network components for Covenant AI Framework with autograd support"
    Version: 1.3.0
    Author: "Karina Mikhailovna Chernykh"
    Rights: "BSD 2-Clause License"
]

;; Import core modules
do %../core/core.reb
do %../core/variable.reb
do %../core/operation-gradients.reb
do %../core/backward.reb
do %../core/autograd.reb
do %../core/central-graph.reb

;; Neural network components with autograd support
nn: context [
    ;; Linear (fully connected) layer - optimized version
    linear: func [
        "Create a linear layer with autograd support"
        input-size [integer!] "Size of input"
        output-size [integer!] "Size of output"
        /requires-grad "Whether to track gradients"
    ] [
        ; Initialize weights and bias with optimized allocation
        weights-data: make block! (output-size * input-size)
        loop (output-size * input-size) [
            append weights-data (random 1.0) * 0.1 - 0.05  ; Small random values
        ]

        bias-data: make block! output-size
        loop output-size [append bias-data 0.0]

        ; Create variables with gradient tracking
        weights-var: variable/requires-grad weights-data either requires-grad [true] [false]
        bias-var: variable/requires-grad bias-data either requires-grad [true] [false]

        make object! [
            type: 'linear
            weights: weights-var
            bias: bias-var
            input-size: input-size
            output-size: output-size

            forward: func [input] [
                ; Matrix multiplication: output = input @ weights^T
                ; Then add bias
                input-times-weights: core/matmul input weights-var
                output-with-bias: core/add input-times-weights bias-var

                output-with-bias
            ]
        ]
    ]

    ;; Linear layer with proper autograd support - optimized version
    linear-grad: func [
        "Create a linear layer with full autograd support"
        input-size [integer!] "Size of input"
        output-size [integer!] "Size of output"
    ] [
        ; Initialize weights with proper autograd support
        weights-data: make block! (output-size * input-size)
        loop (output-size * input-size) [
            append weights-data (random 1.0) * 0.1 - 0.05  ; Small random values
        ]

        bias-data: make block! output-size
        loop output-size [append bias-data 0.0]

        ; Create variables that can track gradients
        weights-var: make object! [
            data: copy weights-data
            shape: reduce [output-size input-size]
            dtype: 'float32
            grad: none
            requires-grad: true
            grad-fn: none
            parents: make block! 0
            children: make block! 0
            is-leaf: true

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]
            ]
        ]

        bias-var: make object! [
            data: copy bias-data
            shape: reduce [output-size]
            dtype: 'float32
            grad: none
            requires-grad: true
            grad-fn: none
            parents: make block! 0
            children: make block! 0
            is-leaf: true

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]
            ]
        ]

        make object! [
            type: 'linear
            weights: weights-var
            bias: bias-var
            input-size: input-size
            output-size: output-size

            forward: func [input] [
                ; Perform linear transformation: y = xW^T + b
                ; Using optimized matrix multiplication
                output-matmul: core/matmul input weights-var
                result: core/add output-matmul bias-var

                result
            ]
        ]
    ]

    ;; ReLU activation function with gradient computation - optimized version
    relu: func [
        "ReLU activation function with gradient tracking"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute ReLU: max(0, x)
        repeat i length? x/data [
            val: x/data/:i
            append result-data either val < 0 [0.0] [val]
        ]

        ; Create gradient function: grad_output where x > 0, else 0
        grad-fn: func [grad-output] [
            grad-input: make block! length? x/data
            repeat i length? x/data [
                if x/data/:i <= 0 [
                    append grad-input 0.0
                ] [
                    append grad-input either i <= length? grad-output [grad-output/:i] [0.0]
                ]
            ]
            reduce [grad-input]
        ]

        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
            parents: reduce [x]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as child to parent
        append x/children result-var

        result-var
    ]

    ;; Leaky ReLU activation function with gradient computation - optimized version
    leaky-relu: func [
        "Leaky ReLU activation function with gradient tracking"
        x [object!]
        /alpha "Negative slope parameter"
        alpha-val [number!] "Alpha value (default 0.01)"
    ] [
        requires-grad: x/requires-grad
        alpha-value: either alpha [alpha-val] [0.01]

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute Leaky ReLU: max(alpha*x, x)
        repeat i length? x/data [
            val: x/data/:i
            append result-data either val < 0 [alpha-value * val] [val]
        ]

        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-input: make block! length? x/data
            repeat i length? x/data [
                if x/data/:i <= 0 [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    append grad-input (grad-val * alpha-value)
                ] [
                    append grad-input either i <= length? grad-output [grad-output/:i] [0.0]
                ]
            ]
            reduce [grad-input]
        ]

        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
            parents: reduce [x]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as child to parent
        append x/children result-var

        result-var
    ]

    ;; Sigmoid activation function with gradient computation - optimized version
    sigmoid: func [
        "Sigmoid activation function with gradient tracking"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute sigmoid: 1 / (1 + e^-x)
        repeat i length? x/data [
            neg-x: 0.0 - x/data/:i
            exp-val: core/exp neg-x
            append result-data (1.0 / (1.0 + exp-val))
        ]

        ; Create gradient function: grad_output * sigmoid(x) * (1 - sigmoid(x))
        grad-fn: func [grad-output] [
            grad-input: make block! length? x/data
            repeat i length? x/data [
                sig-val: result-data/:i
                grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                append grad-input (grad-val * sig-val * (1.0 - sig-val))
            ]
            reduce [grad-input]
        ]

        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
            parents: reduce [x]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as child to parent
        append x/children result-var

        result-var
    ]

    ;; Tanh activation function with autograd support - optimized version
    tanh: func [
        "Tanh activation function with gradient tracking"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute tanh
        repeat i length? x/data [
            exp-pos: core/exp x/data/:i
            exp-neg: core/exp (0.0 - x/data/:i)
            append result-data ((exp-pos - exp-neg) / (exp-pos + exp-neg))
        ]

        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-input: make block! length? x/data
            repeat i length? x/data [
                tanh-val: result-data/:i
                grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                append grad-input (grad-val * (1.0 - (tanh-val * tanh-val)))
            ]
            reduce [grad-input]
        ]

        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
            parents: reduce [x]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as child to parent
        append x/children result-var

        result-var
    ]

    ;; Swish activation function (x * sigmoid(x)) with gradient computation - optimized version
    swish: func [
        "Swish activation function (x * sigmoid(x)) with gradient tracking"
        x [object!]
        /beta "Beta parameter (default 1.0)"
        beta-val [number!] "Beta value for Swish function"
    ] [
        requires-grad: x/requires-grad
        beta-value: either beta [beta-val] [1.0]

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute sigmoid for each element: 1 / (1 + e^(-beta*x))
        sigmoid-values: make block! length? x/data
        repeat i length? x/data [
            neg-x: 0.0 - (beta-value * x/data/:i)
            exp-val: core/exp neg-x
            sig-val: 1.0 / (1.0 + exp-val)
            append sigmoid-values sig-val
            append result-data (x/data/:i * sig-val)
        ]

        ; Create gradient function: grad_output * (swish(x) + sigmoid(x) * (1 - beta * swish(x)))
        grad-fn: func [grad-output] [
            grad-input: make block! length? x/data
            repeat i length? x/data [
                sig-val: sigmoid-values/:i
                swish-val: result-data/:i
                grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                grad-component: swish-val + (sig-val * (1.0 - (beta-value * swish-val)))
                append grad-input (grad-val * grad-component)
            ]
            reduce [grad-input]
        ]

        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
            parents: reduce [x]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as child to parent
        append x/children result-var

        result-var
    ]

    ;; GELU (Gaussian Error Linear Unit) activation function with gradient computation - optimized version
    gelu: func [
        "GELU activation function with gradient tracking"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data

        ; Compute GELU: x * Î¦(x) where Î¦(x) is the cumulative distribution function of the standard normal distribution
        ; We'll approximate Î¦(x) with tanh approximation: 0.5 * (1 + tanh(sqrt(2/Ď€) * (x + 0.044715 * x^3)))
        sqrt-two-over-pi: square-root (2.0 / pi)

        repeat i length? x/data [
            x-val: x/data/:i
            x-cubed: x-val * x-val * x-val
            tanh-arg: sqrt-two-over-pi * (x-val + (0.044715 * x-cubed))
            phi-approx: 0.5 * (1.0 + core/tanh tanh-arg)
            append result-data (x-val * phi-approx)
        ]

        ; Create gradient function: grad_output * (Î¦(x) + x * Ď†(x)) where Ď†(x) is pdf of standard normal
        grad-fn: func [grad-output] [
            grad-input: make block! length? x/data

            ; Precompute constants
            sqrt-two-over-pi: square-root (2.0 / pi)
            sqrt-two-pi: square-root (2.0 * pi)

            repeat i length? x/data [
                x-val: x/data/:i
                x-cubed: x-val * x-val * x-val
                tanh-arg: sqrt-two-over-pi * (x-val + (0.044715 * x-cubed))

                ; Compute Î¦(x) approximation
                phi-approx: 0.5 * (1.0 + core/tanh tanh-arg)

                ; Compute derivative of tanh approximation
                tanh-squared: core/tanh(tanh-arg) * core/tanh(tanh-arg)
                phi-prime-approx: 0.5 * sqrt-two-over-pi * (1.0 - tanh-squared) * (1.0 + (0.044715 * 3 * x-val * x-val))

                ; GELU derivative: Î¦(x) + x * Ď†(x) â‰� Î¦(x) + x * phi_prime_approx
                gelu-deriv: phi-approx + (x-val * phi-prime-approx)

                grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                append grad-input (grad-val * gelu-deriv)
            ]
            reduce [grad-input]
        ]

        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
            parents: reduce [x]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as child to parent
        append x/children result-var

        result-var
    ]

    ;; Softmax activation function with gradient computation - optimized version
    softmax: func [
        "Softmax activation function with gradient tracking"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate data
        result-data: make block! length? x/data

        ; Compute softmax: exp(x_i) / sum(exp(x_j))
        ; First subtract max for numerical stability
        max-val: first x/data
        foreach val next x/data [if val > max-val [max-val: val]]

        exp-data: make block! length? x/data
        repeat i length? x/data [
            append exp-data core/exp(x/data/:i - max-val)
        ]

        sum-exp: 0.0
        foreach val exp-data [sum-exp: sum-exp + val]

        ; Compute final softmax values
        repeat i length? exp-data [
            append result-data (exp-data/:i / sum-exp)
        ]

        ; Create gradient function for softmax
        grad-fn: func [grad-output] [
            grad-input: make block! length? x/data
            loop length? x/data [append grad-input 0.0]

            ; Compute Jacobian-vector product for softmax
            repeat i length? result-data [
                sum-jacobian: 0.0
                repeat j length? result-data [
                    kronecker-delta: either i = j [1.0] [0.0]
                    jacobian-element: result-data/:i * (kronecker-delta - result-data/:j)
                    grad-out-val: either j <= length? grad-output [grad-output/:j] [0.0]
                    sum-jacobian: sum-jacobian + (jacobian-element * grad-out-val)
                ]
                grad-input/:i: sum-jacobian
            ]

            reduce [grad-input]
        ]

        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
            parents: reduce [x]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as child to parent
        append x/children result-var

        result-var
    ]

    ;; Mean squared error loss with gradient computation - optimized version
    mse-loss: func [
        "Mean squared error loss with gradient tracking"
        predictions [object!]
        targets [object!]
    ] [
        ; Pre-allocate data
        diff-data: make block! length? predictions/data
        ; Compute (predictions - targets)^2
        repeat i length? predictions/data [
            pred-val: predictions/data/:i
            target-val: either i <= length? targets/data [targets/data/:i] [0.0]
            diff: pred-val - target-val
            append diff-data (diff * diff)
        ]

        ; Compute mean
        n: length? diff-data
        if n = 0 [n: 1]  ; Prevent division by zero
        sum: 0.0
        foreach val diff-data [sum: sum + val]
        mean-val: sum / n

        ; Create result variable
        result-var: make object! [
            data: reduce [mean-val]
            shape: reduce [1]
            dtype: 'float32
            grad: none
            requires-grad: predictions/requires-grad
            grad-fn: none
            parents: reduce [predictions targets]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as children to parents
        append predictions/children result-var
        append targets/children result-var

        ; Set up gradient function for MSE
        result-var/grad-fn: func [grad-output] [
            ; Gradient w.r.t. predictions: 2*(pred - target)/n
            grad-pred: make block! length? predictions/data
            grad-target: make block! length? targets/data

            grad-scaler: (grad-output/1 * 2.0) / n
            repeat i length? predictions/data [
                target-val: either i <= length? targets/data [targets/data/:i] [0.0]
                pred-val: predictions/data/:i
                append grad-pred ((pred-val - target-val) * grad-scaler)
            ]

            ; Gradient w.r.t. targets: -2*(pred - target)/n
            repeat i length? targets/data [
                pred-val: either i <= length? predictions/data [predictions/data/:i] [0.0]
                target-val: targets/data/:i
                append grad-target (((pred-val - target-val) * -1.0) * grad-scaler)
            ]

            reduce [grad-pred grad-target]
        ]

        result-var
    ]

    ;; Cross entropy loss with gradient computation - optimized version
    cross-entropy-loss: func [
        "Cross entropy loss with gradient tracking"
        predictions [object!]
        targets [object!]
    ] [
        ; Pre-allocate data
        log-preds: make block! length? predictions/data
        ; Compute cross entropy: -sum(targets * log(predictions))
        repeat i length? predictions/data [
            pred-val: predictions/data/:i
            log-val: core/log(pred-val + 1e-8)  ; Add small epsilon to prevent log(0)
            append log-preds log-val
        ]

        result-data: make block! length? targets/data
        repeat i length? targets/data [
            target-val: targets/data/:i
            log-val: either i <= length? log-preds [log-preds/:i] [0.0]
            append result-data (0.0 - (target-val * log-val))
        ]

        ; Sum all elements
        total: 0.0
        foreach val result-data [total: total + val]
        mean-val: total / length? result-data

        ; Create result variable
        result-var: make object! [
            data: reduce [mean-val]
            shape: reduce [1]
            dtype: 'float32
            grad: none
            requires-grad: predictions/requires-grad
            grad-fn: none
            parents: reduce [predictions targets]
            children: make block! 0
            is-leaf: false

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]

                if self/grad-fn [
                    parent-grads: self/grad-fn grad-data
                    repeat i length? self/parents [
                        parent: self/parents/:i
                        parent-grad: parent-grads/:i
                        parent/backward/gradient parent-grad
                    ]
                ]
            ]
        ]

        ; Add as children to parents
        append predictions/children result-var
        append targets/children result-var

        ; Set up gradient function for cross entropy
        result-var/grad-fn: func [grad-output] [
            ; Gradient w.r.t. predictions: -targets / (predictions + epsilon)
            grad-pred: make block! length? predictions/data
            grad-target: make block! length? targets/data

            grad-scaler: grad-output/1 / length? predictions/data
            repeat i length? predictions/data [
                pred-val: predictions/data/:i
                target-val: either i <= length? targets/data [targets/data/:i] [0.0]
                append grad-pred (0.0 - (target-val / (pred-val + 1e-8)) * grad-scaler)
            ]

            ; Gradient w.r.t. targets: -log(predictions)
            repeat i length? targets/data [
                pred-val: either i <= length? predictions/data [predictions/data/:i] [1.0]
                log-pred: core/log(pred-val + 1e-8)
                append grad-target (0.0 - log-pred * grad-scaler)
            ]

            reduce [grad-pred grad-target]
        ]

        result-var
    ]

    ;; Convolutional layer (simplified) - optimized version
    conv1d: func [
        "Create a 1D convolutional layer with autograd support"
        in-channels [integer!] "Number of input channels"
        out-channels [integer!] "Number of output channels"
        kernel-size [integer!] "Size of the convolution kernel"
    ] [
        ; Initialize weights and bias
        weights-data: make block! (out-channels * in-channels * kernel-size)
        loop (out-channels * in-channels * kernel-size) [
            append weights-data (random 1.0) * 0.1 - 0.05  ; Small random values
        ]

        bias-data: make block! out-channels
        loop out-channels [append bias-data 0.0]

        ; Create variables that can track gradients
        weights-var: make object! [
            data: copy weights-data
            shape: reduce [out-channels in-channels kernel-size]
            dtype: 'float32
            grad: none
            requires-grad: true
            grad-fn: none
            parents: make block! 0
            children: make block! 0
            is-leaf: true

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]
            ]
        ]

        bias-var: make object! [
            data: copy bias-data
            shape: reduce [out-channels]
            dtype: 'float32
            grad: none
            requires-grad: true
            grad-fn: none
            parents: make block! 0
            children: make block! 0
            is-leaf: true

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]
            ]
        ]

        make object! [
            type: 'conv1d
            weights: weights-var
            bias: bias-var
            in-channels: in-channels
            out-channels: out-channels
            kernel-size: kernel-size

            forward: func [input] [
                ; Simplified 1D convolution implementation
                ; input shape: [sequence-length, in-channels]
                seq-len: either length? input/shape > 1 [input/shape/1] [length? input/data]

                ; Calculate output sequence length (assuming no padding and stride=1)
                out-seq-len: seq-len - kernel-size + 1

                ; Pre-initialize output data
                output-data: make block! (out-seq-len * out-channels)
                loop (out-seq-len * out-channels) [append output-data 0.0]

                ; Perform convolution with optimized access
                repeat out-chan out-channels [
                    repeat out-pos out-seq-len [
                        sum: 0.0
                        repeat k-kernel kernel-size [
                            repeat in-chan in-channels [
                                ; Calculate indices
                                input-pos: out-pos + k-kernel - 1

                                ; Calculate input data index
                                input-idx: ((input-pos - 1) * in-channels) + in-chan
                                weight-idx: (((out-chan - 1) * in-channels) * kernel-size) + ((in-chan - 1) * kernel-size) + k-kernel

                                ; Add contribution to sum
                                if all [input-idx <= length? input/data weight-idx <= length? weights-var/data] [
                                    sum: sum + (input/data/:input-idx * weights-var/data/:weight-idx)
                                ]
                            ]
                        ]

                        ; Add bias
                        sum: sum + bias-var/data/:out-chan

                        ; Store result
                        output-idx: ((out-chan - 1) * out-seq-len) + out-pos
                        output-data/:output-idx: sum
                    ]
                ]

                ; Create output variable
                output-var: make object! [
                    data: output-data
                    shape: reduce [out-seq-len out-channels]
                    dtype: 'float32
                    grad: none
                    requires-grad: input/requires-grad or weights-var/requires-grad
                    grad-fn: none
                    parents: reduce [input weights-var bias-var]
                    children: make block! 0
                    is-leaf: false

                    set-requires-grad: func [flag [logic!]] [
                        self/requires-grad: flag
                        if flag and not self/grad [
                            self/grad: make block! length? self/data
                            loop length? self/data [append self/grad 0.0]
                        ]
                    ]

                    zero-grad: func [] [
                        if self/grad [
                            repeat i length? self/grad [
                                self/grad/:i: 0.0
                            ]
                        ]
                    ]

                    backward: func [/gradient grad-data [block!]] [
                        if not gradient [
                            grad-data: make block! length? self/data
                            loop length? self/data [append grad-data 1.0]
                        ]

                        if self/requires-grad [
                            if not self/grad [
                                self/grad: make block! length? self/data
                                loop length? self/data [append self/grad 0.0]
                            ]

                            repeat i length? self/grad [
                                if i <= length? grad-data [
                                    self/grad/:i: self/grad/:i + grad-data/:i
                                ]
                            ]
                        ]

                        if self/grad-fn [
                            parent-grads: self/grad-fn grad-data
                            repeat i length? self/parents [
                                parent: self/parents/:i
                                parent-grad: parent-grads/:i
                                parent/backward/gradient parent-grad
                            ]
                        ]
                    ]
                ]

                ; Add as children to parents
                append input/children output-var
                append weights-var/children output-var
                append bias-var/children output-var

                ; Set up gradient function for convolution
                output-var/grad-fn: func [grad-output] [
                    ; Gradient w.r.t. input
                    grad-input: make block! length? input/data
                    loop length? input/data [append grad-input 0.0]

                    ; Gradient w.r.t. weights
                    grad-weights: make block! length? weights-var/data
                    loop length? weights-var/data [append grad-weights 0.0]

                    ; Gradient w.r.t. bias
                    grad-bias: make block! length? bias-var/data
                    loop length? bias-var/data [append grad-bias 0.0]

                    ; Compute gradients
                    repeat out-chan out-channels [
                        repeat out-pos out-seq-len [
                            grad-val: grad-output/(((out-chan - 1) * out-seq-len) + out-pos)

                            ; Update bias gradient
                            grad-bias/:out-chan: grad-bias/:out-chan + grad-val

                            repeat k-kernel kernel-size [
                                repeat in-chan in-channels [
                                    ; Calculate indices
                                    input-pos: out-pos + k-kernel - 1

                                    ; Calculate input data index
                                    input-idx: ((input-pos - 1) * in-channels) + in-chan
                                    weight-idx: (((out-chan - 1) * in-channels) * kernel-size) + ((in-chan - 1) * kernel-size) + k-kernel

                                    ; Update gradients
                                    if all [input-idx <= length? input/data weight-idx <= length? weights-var/data] [
                                        grad-input/:input-idx: grad-input/:input-idx + (grad-val * weights-var/data/:weight-idx)
                                        grad-weights/:weight-idx: grad-weights/:weight-idx + (grad-val * input/data/:input-idx)
                                    ]
                                ]
                            ]
                        ]
                    ]

                    reduce [grad-input grad-weights grad-bias]
                ]

                output-var
            ]
        ]
    ]

    ;; Sequential module for chaining layers - optimized version
    sequential: func [
        "Sequential module for chaining layers"
        layers [block!] "Block of layers to chain"
    ] [
        make object! [
            type: 'sequential
            layers: layers

            forward: func [input] [
                result: input
                foreach layer layers [
                    result: layer/forward result
                ]
                result
            ]
        ]
    ]

    ;; Dropout layer - optimized version
    dropout: func [
        "Dropout layer with probability p"
        /prob "Dropout probability"
        p [number!] "Probability (default 0.5)"
    ] [
        dropout-prob: either prob [p] [0.5]

        make object! [
            type: 'dropout
            p: dropout-prob

            forward: func [input train [logic!]] [
                if not train [return input]  ; Don't apply dropout during inference

                ; Pre-allocate mask
                mask: make block! length? input/data
                repeat i length? input/data [
                    if random 1.0 > dropout-prob [
                        append mask 1.0 / (1.0 - dropout-prob)  ; Scale up to maintain expected value
                    ] [
                        append mask 0.0
                    ]
                ]

                ; Element-wise multiply input with mask
                result-data: make block! length? input/data
                repeat i length? input/data [
                    append result-data (input/data/:i * mask/:i)
                ]

                ; Create result variable
                result-var: make object! [
                    data: result-data
                    shape: input/shape
                    dtype: input/dtype
                    grad: none
                    requires-grad: input/requires-grad
                    grad-fn: none
                    parents: reduce [input]
                    children: make block! 0
                    is-leaf: false

                    set-requires-grad: func [flag [logic!]] [
                        self/requires-grad: flag
                        if flag and not self/grad [
                            self/grad: make block! length? self/data
                            loop length? self/data [append self/grad 0.0]
                        ]
                    ]

                    zero-grad: func [] [
                        if self/grad [
                            repeat i length? self/grad [
                                self/grad/:i: 0.0
                            ]
                        ]
                    ]

                    backward: func [/gradient grad-data [block!]] [
                        if not gradient [
                            grad-data: make block! length? self/data
                            loop length? self/data [append grad-data 1.0]
                        ]

                        if self/requires-grad [
                            if not self/grad [
                                self/grad: make block! length? self/data
                                loop length? self/data [append self/grad 0.0]
                            ]

                            repeat i length? self/grad [
                                if i <= length? grad-data [
                                    self/grad/:i: self/grad/:i + grad-data/:i
                                ]
                            ]
                        ]

                        if self/grad-fn [
                            parent-grads: self/grad-fn grad-data
                            repeat i length? self/parents [
                                parent: self/parents/:i
                                parent-grad: parent-grads/:i
                                parent/backward/gradient parent-grad
                            ]
                        ]
                    ]
                ]

                ; Add as child to parent
                append input/children result-var

                ; Set up gradient function
                result-var/grad-fn: func [grad-output] [
                    grad-input: make block! length? input/data
                    repeat i length? input/data [
                        grad-out-val: either i <= length? grad-output [grad-output/:i] [0.0]
                        append grad-input (grad-out-val * mask/:i)
                    ]
                    reduce [grad-input]
                ]

                result-var
            ]
        ]
    ]

    ;; Batch normalization 1D - optimized version
    batchnorm1d: func [
        "Batch normalization 1D layer"
        num-features [integer!] "Number of features"
    ] [
        ; Initialize running statistics
        running-mean: make block! num-features
        loop num-features [append running-mean 0.0]

        running-var: make block! num-features
        loop num-features [append running-var 1.0]  ; Start with variance 1

        ; Initialize learnable parameters
        gamma-data: make block! num-features
        loop num-features [append gamma-data 1.0]  ; Start with gamma = 1

        beta-data: make block! num-features
        loop num-features [append beta-data 0.0]   ; Start with beta = 0

        ; Create variables that can track gradients
        gamma-var: make object! [
            data: copy gamma-data
            shape: reduce [num-features]
            dtype: 'float32
            grad: none
            requires-grad: true
            grad-fn: none
            parents: make block! 0
            children: make block! 0
            is-leaf: true

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]
            ]
        ]

        beta-var: make object! [
            data: copy beta-data
            shape: reduce [num-features]
            dtype: 'float32
            grad: none
            requires-grad: true
            grad-fn: none
            parents: make block! 0
            children: make block! 0
            is-leaf: true

            set-requires-grad: func [flag [logic!]] [
                self/requires-grad: flag
                if flag and not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
            ]

            zero-grad: func [] [
                if self/grad [
                    repeat i length? self/grad [
                        self/grad/:i: 0.0
                    ]
                ]
            ]

            backward: func [/gradient grad-data [block!]] [
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]

                if self/requires-grad [
                    if not self/grad [
                        self/grad: make block! length? self/data
                        loop length? self/data [append self/grad 0.0]
                    ]

                    repeat i length? self/grad [
                        if i <= length? grad-data [
                            self/grad/:i: self/grad/:i + grad-data/:i
                        ]
                    ]
                ]
            ]
        ]

        make object! [
            type: 'batchnorm1d
            gamma: gamma-var
            beta: beta-var
            running_mean: copy running-mean
            running_var: copy running-var
            num_features: num-features
            momentum: 0.1
            eps: 1e-5
            training: true

            forward: func [input] [
                batch-size: either length? input/shape > 1 [input/shape/1] [1]

                ; Pre-allocate batch statistics
                batch-mean: make block! num-features
                batch-var: make block! num-features

                if training [
                    ; Calculate mean for each feature
                    repeat feat num-features [
                        sum: 0.0
                        repeat b batch-size [
                            idx: ((b - 1) * num-features) + feat
                            sum: sum + input/data/:idx
                        ]
                        append batch-mean (sum / batch-size)
                    ]

                    ; Calculate variance for each feature
                    repeat feat num-features [
                        sum: 0.0
                        mean-val: batch-mean/:feat
                        repeat b batch-size [
                            idx: ((b - 1) * num-features) + feat
                            diff: input/data/:idx - mean-val
                            sum: sum + (diff * diff)
                        ]
                        append batch-var (sum / batch-size)
                    ]

                    ; Update running statistics
                    repeat feat num-features [
                        running-mean/:feat: (momentum * batch-mean/:feat) + ((1 - momentum) * running-mean/:feat)
                        running-var/:feat: (momentum * batch-var/:feat) + ((1 - momentum) * running-var/:feat)
                    ]
                ] [
                    ; Use running statistics during evaluation
                    batch-mean: copy running-mean
                    batch-var: copy running-var
                ]

                ; Normalize
                norm-data: make block! length? input/data
                repeat i length? input/data [
                    feat-idx: ((i - 1) // num-features) + 1
                    val: input/data/:i
                    mean: batch-mean/:feat-idx
                    var: batch-var/:feat-idx
                    append norm-data ((val - mean) / core/square-root(var + eps))
                ]

                ; Scale and shift
                result-data: make block! length? norm-data
                repeat i length? norm-data [
                    feat-idx: ((i - 1) // num-features) + 1
                    norm-val: norm-data/:i
                    gamma-val: gamma-var/data/:feat-idx
                    beta-val: beta-var/data/:feat-idx
                    append result-data ((norm-val * gamma-val) + beta-val)
                ]

                ; Create result variable
                result-var: make object! [
                    data: result-data
                    shape: input/shape
                    dtype: input/dtype
                    grad: none
                    requires-grad: input/requires-grad or gamma-var/requires-grad or beta-var/requires-grad
                    grad-fn: none
                    parents: reduce [input gamma-var beta-var]
                    children: make block! 0
                    is-leaf: false

                    set-requires-grad: func [flag [logic!]] [
                        self/requires-grad: flag
                        if flag and not self/grad [
                            self/grad: make block! length? self/data
                            loop length? self/data [append self/grad 0.0]
                        ]
                    ]

                    zero-grad: func [] [
                        if self/grad [
                            repeat i length? self/grad [
                                self/grad/:i: 0.0
                            ]
                        ]
                    ]

                    backward: func [/gradient grad-data [block!]] [
                        if not gradient [
                            grad-data: make block! length? self/data
                            loop length? self/data [append grad-data 1.0]
                        ]

                        if self/requires-grad [
                            if not self/grad [
                                self/grad: make block! length? self/data
                                loop length? self/data [append self/grad 0.0]
                            ]

                            repeat i length? self/grad [
                                if i <= length? grad-data [
                                    self/grad/:i: self/grad/:i + grad-data/:i
                                ]
                            ]
                        ]

                        if self/grad-fn [
                            parent-grads: self/grad-fn grad-data
                            repeat i length? self/parents [
                                parent: self/parents/:i
                                parent-grad: parent-grads/:i
                                parent/backward/gradient parent-grad
                            ]
                        ]
                    ]
                ]

                ; Add as children to parents
                append input/children result-var
                append gamma-var/children result-var
                append beta-var/children result-var

                ; Set up gradient function for batch norm
                result-var/grad-fn: func [grad-output] [
                    ; Gradients for gamma, beta, and input
                    grad-gamma: make block! length? gamma-var/data
                    grad-beta: make block! length? beta-var/data
                    grad-input: make block! length? input/data

                    loop length? gamma-var/data [append grad-gamma 0.0]
                    loop length? beta-var/data [append grad-beta 0.0]
                    loop length? input/data [append grad-input 0.0]

                    ; Calculate gradients
                    repeat feat num-features [
                        ; Gradient w.r.t. beta
                        sum: 0.0
                        repeat b batch-size [
                            idx: ((b - 1) * num-features) + feat
                            grad-out-val: either idx <= length? grad-output [grad-output/:idx] [0.0]
                            sum: sum + grad-out-val
                        ]
                        grad-beta/:feat: sum

                        ; Gradient w.r.t. gamma
                        sum: 0.0
                        repeat b batch-size [
                            idx: ((b - 1) * num-features) + feat
                            norm-val: norm-data/:idx
                            grad-out-val: either idx <= length? grad-output [grad-output/:idx] [0.0]
                            sum: sum + (grad-out-val * norm-val)
                        ]
                        grad-gamma/:feat: sum

                        ; Gradient w.r.t. input
                        mean: batch-mean/:feat
                        var: batch-var/:feat
                        std: core/square-root(var + eps)

                        repeat b batch-size [
                            idx: ((b - 1) * num-features) + feat
                            grad-out: either idx <= length? grad-output [grad-output/:idx] [0.0]
                            norm-val: norm-data/:idx
                            gamma-val: gamma-var/data/:feat

                            ; Combined gradient
                            grad-input/:idx: (gamma-val / std) * (grad-out - (norm-val * grad-gamma/:feat + grad-beta/:feat) / batch-size)
                        ]
                    ]

                    reduce [grad-input grad-gamma grad-beta]
                ]

                result-var
            ]

            train: func [] [training: true]
            eval: func [] [training: false]
        ]
    ]

]
