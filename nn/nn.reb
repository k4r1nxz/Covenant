REBOL [
    Title: "Covenant Neural Network Module"
    Description: "Neural network components for Covenant AI Framework with autograd support"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
    Rights: "BSD 2-Clause License"
]

;; Import core modules
do %../core/core.reb
do %../core/variable.reb
do %../core/operation-gradients.reb
do %../core/backward.reb
do %../core/central-graph.reb

;; Neural network components with autograd support
nn: context [
    ;; Linear (fully connected) layer
    linear: func [
        "Create a linear layer with autograd support"
        input-size [integer!] "Size of input"
        output-size [integer!] "Size of output"
        /requires-grad "Whether to track gradients"
    ] [
        ; Initialize weights and bias
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
                
                ; First transpose weights (this is a simplified approach)
                ; In a real implementation, we'd properly handle the transpose
                transposed-w: copy weights-var/data
                ; For now, we'll just use the weights as-is for the matmul
                
                ; Perform matrix multiplication using the enhanced autograd ops
                input-times-weights: enhanced-ops/mul input weights-var
                output-with-bias: enhanced-ops/add input-times-weights bias-var
                
                output-with-bias
            ]
        ]
    ]
    
    ;; Linear layer with proper autograd support
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
                ; Using central graph operations
                ; Matrix multiplication: input @ weights^T
                rows-in: either length? input/shape > 1 [input/shape/1] [1]
                cols-in: input/shape/1
                if length? input/shape > 1 [cols-in: input/shape/2]

                rows-w: weights-var/shape/1
                cols-w: weights-var/shape/2

                ; Compute output dimensions
                out-rows: rows-in
                out-cols: rows-w  ; Because we're doing input @ weights^T

                output-data: make block! (out-rows * out-cols)
                loop (out-rows * out-cols) [append output-data 0.0]

                ; Perform matrix multiplication
                repeat i out-rows [
                    repeat j out-cols [
                        sum: 0.0
                        repeat k cols-in [
                            in-idx: ((i - 1) * cols-in) + k
                            w-idx: ((j - 1) * cols-w) + k  ; Transposed index
                            if all [in-idx <= length? input/data w-idx <= length? weights-var/data] [
                                sum: sum + (input/data/:in-idx * weights-var/data/:w-idx)
                            ]
                        ]
                        out-idx: ((i - 1) * out-cols) + j
                        output-data/:out-idx: sum
                    ]
                ]

                ; Create output variable using central graph
                output-var: make object! [
                    data: output-data
                    shape: reduce [out-rows out-cols]
                    dtype: 'float32
                    grad: none
                    requires-grad: input/requires-grad or weights-var/requires-grad
                    grad-fn: none
                    parents: reduce [input weights-var]
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

                        ; Use central graph manager for backward pass
                        graph-manager/backward-pass/self grad-data
                    ]
                ]

                ; Register with central graph manager
                graph-manager/add-node output-var
                append input/children output-var
                append weights-var/children output-var

                ; Add bias using central graph operations
                result-data: copy output-var/data
                bias-len: length? bias-var/data
                repeat i length? result-data [
                    bias-idx: ((i - 1) // bias-len) + 1  ; Broadcast bias
                    result-data/:i: result-data/:i + bias-var/data/:bias-idx
                ]

                ; Create final result variable using central graph
                result-var: make object! [
                    data: result-data
                    shape: output-var/shape
                    dtype: 'float32
                    grad: none
                    requires-grad: output-var/requires-grad or bias-var/requires-grad
                    grad-fn: none
                    parents: reduce [output-var bias-var]
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

                        ; Use central graph manager for backward pass
                        graph-manager/backward-pass/self grad-data
                    ]
                ]

                ; Register with central graph manager
                graph-manager/add-node result-var
                graph-manager/register-output result-var
                append output-var/children result-var
                append bias-var/children result-var

                ; Set up gradient functions for the operations
                ; For matrix multiplication
                output-var/grad-fn: func [grad-output] [
                    ; Gradient w.r.t. input: grad_output @ weights
                    rows-out-g: either length? output-var/shape > 1 [output-var/shape/1] [1]
                    cols-out-g: output-var/shape/1
                    if length? output-var/shape > 1 [cols-out-g: output-var/shape/2]

                    rows-w: weights-var/shape/1
                    cols-w: weights-var/shape/2

                    grad-input: make block! (rows-out-g * cols-w)
                    loop (rows-out-g * cols-w) [append grad-input 0.0]

                    repeat i rows-out-g [
                        repeat j cols-w [
                            sum: 0.0
                            repeat k rows-w [
                                grad-out-idx: ((i - 1) * cols-out-g) + k
                                w-idx: ((k - 1) * cols-w) + j  ; Transposed index
                                if all [grad-out-idx <= length? grad-output w-idx <= length? weights-var/data] [
                                    sum: sum + (grad-output/:grad-out-idx * weights-var/data/:w-idx)
                                ]
                            ]
                            in-idx: ((i - 1) * cols-w) + j
                            grad-input/:in-idx: sum
                        ]
                    ]

                    ; Gradient w.r.t. weights: input^T @ grad_output
                    grad-weights: make block! (rows-w * cols-w)
                    loop (rows-w * cols-w) [append grad-weights 0.0]

                    repeat i rows-w [
                        repeat j cols-w [
                            sum: 0.0
                            repeat k rows-out-g [
                                in-idx: ((k - 1) * cols-w) + i  ; Transposed index
                                grad-out-idx: ((k - 1) * cols-out-g) + j
                                if all [in-idx <= length? input/data grad-out-idx <= length? grad-output] [
                                    sum: sum + (input/data/:in-idx * grad-output/:grad-out-idx)
                                ]
                            ]
                            w-idx: ((i - 1) * cols-w) + j
                            grad-weights/:w-idx: sum
                        ]
                    ]

                    reduce [grad-input grad-weights]
                ]

                ; For bias addition
                result-var/grad-fn: func [grad-output] [
                    ; Gradient w.r.t. output (from matmul)
                    grad-output-from-matmul: copy grad-output

                    ; Gradient w.r.t. bias is the same as the output gradient (summed across batches if needed)
                    grad-bias: make block! length? bias-var/data
                    loop length? bias-var/data [append grad-bias 0.0]

                    ; Sum gradients across batch dimension for bias
                    out-rows: either length? result-var/shape > 1 [result-var/shape/1] [1]
                    out-cols: result-var/shape/1
                    if length? result-var/shape > 1 [out-cols: result-var/shape/2]

                    repeat j length? bias-var/data [
                        sum: 0.0
                        repeat i out-rows [
                            grad-out-idx: ((i - 1) * out-cols) + j
                            if grad-out-idx <= length? grad-output [
                                sum: sum + grad-output/:grad-out-idx
                            ]
                        ]
                        grad-bias/:j: sum
                    ]

                    reduce [grad-output-from-matmul grad-bias]
                ]

                result-var
            ]
        ]
    ]
    
    ;; ReLU activation function with central graph support
    relu: func [
        "ReLU activation function with gradient tracking"
        x [object!]
    ] [
        ; Use the central graph ReLU operation
        central-ops/relu x
    ]

    ;; Sigmoid activation function with central graph support
    sigmoid: func [
        "Sigmoid activation function with gradient tracking"
        x [object!]
    ] [
        ; Use the central graph sigmoid operation
        central-ops/sigmoid x
    ]
    
    ;; Tanh activation function with autograd support
    tanh: func [
        "Tanh activation function with gradient tracking"
        x [object!]
    ] [
        ; Create a variable for the tanh operation
        requires-grad: x/requires-grad
        
        ; Compute tanh
        result-data: copy x/data
        repeat i length? result-data [
            exp-pos: exp(result-data/:i)
            exp-neg: exp(0.0 - result-data/:i)
            result-data/:i: (exp-pos - exp-neg) / (exp-pos + exp-neg)
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-input: copy grad-output
            repeat i length? grad-input [
                if i <= length? result-data [
                    tanh-val: result-data/:i
                    grad-input/:i: grad-input/:i * (1.0 - (tanh-val * tanh-val))
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
    
    ;; Mean squared error loss with central graph support
    mse-loss: func [
        "Mean squared error loss with gradient tracking"
        predictions [object!]
        targets [object!]
    ] [
        ; Compute (predictions - targets)^2
        diff-data: copy predictions/data
        repeat i length? diff-data [
            if i <= length? targets/data [
                diff-data/:i: diff-data/:i - targets/data/:i
            ]
        ]

        squared-data: copy diff-data
        repeat i length? squared-data [
            squared-data/:i: squared-data/:i * squared-data/:i
        ]

        ; Compute mean
        n: length? squared-data
        if n = 0 [n: 1]  ; Prevent division by zero
        sum: 0.0
        foreach val squared-data [sum: sum + val]
        mean-val: sum / n

        ; Create result variable using central graph
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

                ; Use central graph manager for backward pass
                graph-manager/backward-pass/self grad-data
            ]
        ]

        ; Register with central graph manager
        graph-manager/add-node result-var
        graph-manager/register-output result-var
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
    
    ;; Convolutional layer (simplified)
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
                ; Simplified convolution implementation
                ; In a real implementation, we'd properly handle the convolution operation
                print "Conv1d forward pass - simplified implementation"
                
                ; For now, return a dummy result that maintains gradient tracking
                result-data: make block! out-channels
                loop out-channels [append result-data 0.0]
                
                make object! [
                    data: result-data
                    shape: reduce [out-channels]
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
            ]
        ]
    ]
    
    ;; Dropout layer (simplified)
    dropout: func [
        "Create a dropout layer with autograd support"
        p [number!] "Dropout probability"
    ] [
        make object! [
            type: 'dropout
            p: p
            training: true
            
            forward: func [input] [
                if training [
                    ; Generate random mask
                    mask-data: make block! length? input/data
                    repeat i length? input/data [
                        if random 1.0 < p [
                            append mask-data 0.0
                        ] [
                            append mask-data (1.0 / (1.0 - p))  ; Scale up non-dropped values
                        ]
                    ]
                    
                    ; Apply mask
                    result-data: copy input/data
                    repeat i length? result-data [
                        result-data/:i: result-data/:i * mask-data/:i
                    ]
                    
                    ; Return a new variable that can track gradients
                    make object! [
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
                            
                            ; For dropout, gradient is just masked
                            input-grad: copy grad-data
                            repeat i length? input-grad [
                                input-grad/:i: input-grad/:i * either i <= length? mask-data [mask-data/:i] [1.0]
                            ]
                            
                            input/backward/gradient input-grad
                        ]
                    ]
                ] [
                    ; In evaluation mode, just return input
                    input
                ]
            ]
            
            train: func [] [training: true]
            eval: func [] [training: false]
        ]
    ]
    
    ;; Softmax activation
    softmax: func [
        "Softmax activation function with gradient tracking"
        x [object!]
    ] [
        ; Find max for numerical stability
        max-val: first x/data
        foreach val x/data [
            if val > max-val [max-val: val]
        ]
        
        ; Subtract max and exponentiate
        exp-data: copy x/data
        repeat i length? exp-data [
            exp-data/:i: exp(x/data/:i - max-val)
        ]
        
        ; Sum exponentials
        sum-exp: 0.0
        foreach val exp-data [sum-exp: sum-exp + val]
        
        ; Divide by sum
        result-data: copy exp-data
        repeat i length? result-data [
            result-data/:i: result-data/:i / sum-exp
        ]
        
        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: x/shape
            dtype: x/dtype
            grad: none
            requires-grad: x/requires-grad
            grad-fn: none
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
        
        ; Set up gradient function for softmax
        result-var/grad-fn: func [grad-output] [
            ; Compute Jacobian of softmax
            n: length? result-data
            jacobian: make block! (n * n)
            loop (n * n) [append jacobian 0.0]
            
            ; Fill the Jacobian matrix
            repeat i n [
                repeat j n [
                    idx: ((i - 1) * n) + j
                    if i = j [
                        jacobian/:idx: result-data/:i * (1.0 - result-data/:j)
                    ] [
                        jacobian/:idx: 0.0 - (result-data/:i * result-data/:j)
                    ]
                ]
            ]
            
            ; Compute gradient: grad_input = grad_output * jacobian
            grad-input: make block! n
            loop n [append grad-input 0.0]
            
            repeat i n [
                sum: 0.0
                repeat j n [
                    grad-out-idx: j
                    jac-idx: ((i - 1) * n) + j
                    if all [grad-out-idx <= length? grad-output jac-idx <= length? jacobian] [
                        sum: sum + (grad-output/:grad-out-idx * jacobian/:jac-idx)
                    ]
                ]
                grad-input/:i: sum
            ]
            
            reduce [grad-input]
        ]
        
        ; Add as child to parent
        append x/children result-var
        
        result-var
    ]
    
    ;; Helper function for exponentiation (for sigmoid/tanh)
    exp: func [
        "Calculate e^x"
        x [number!]
    ] [
        ; Using a simple approximation for e^x
        ; For a production implementation, this would use a more accurate method
        result: 1.0  ; First term is 1
        term: 1.0     ; Current term in the series
        
        ; Use first 9 additional terms of Taylor series (so 10 total): e^x = 1 + x + x^2/2! + x^3/3! + ...
        repeat n 9 [
            term: term * x / n  ; This correctly calculates x^n/n! incrementally
            result: result + term
        ]
        result
    ]
]