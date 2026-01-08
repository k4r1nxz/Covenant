REBOL [
    Title: "Covenant Operation Gradients"
    Description: "Gradient computation for various operations in Covenant AI Framework"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Import necessary modules
do %variable.reb

;; Gradient computation for operations
operation-gradients: context [
    ;; Identity operation gradient
    identity-grad: func [grad-output] [
        grad-output
    ]
    
    ;; Addition operation gradient
    add-grad: func [grad-output] [
        ; Gradient flows equally to both inputs
        reduce [grad-output grad-output]
    ]
    
    ;; Subtraction operation gradient
    sub-grad: func [grad-output] [
        ; Gradient for first input is positive, for second is negative
        grad-a: copy grad-output
        grad-b: copy grad-output
        repeat i length? grad-b [
            grad-b/:i: 0.0 - grad-b/:i  ; Negative gradient for second operand
        ]
        reduce [grad-a grad-b]
    ]
    
    ;; Multiplication operation gradient
    mul-grad: func [a-data b-data grad-output] [
        ; d/dx (x*y) = y, d/dy (x*y) = x
        grad-a: copy grad-output
        grad-b: copy grad-output
        
        ; Multiply by the other operand
        repeat i length? grad-a [
            if i <= length? b-data [
                grad-a/:i: grad-a/:i * b-data/:i
            ]
        ]
        
        repeat i length? grad-b [
            if i <= length? a-data [
                grad-b/:i: grad-b/:i * a-data/:i
            ]
        ]
        
        reduce [grad-a grad-b]
    ]
    
    ;; Division operation gradient
    div-grad: func [a-data b-data grad-output] [
        ; d/dx (x/y) = 1/y, d/dy (x/y) = -x/y^2
        grad-a: copy grad-output
        grad-b: copy grad-output
        
        repeat i length? grad-a [
            if all [i <= length? b-data b-data/:i <> 0.0] [
                grad-a/:i: grad-a/:i / b-data/:i
            ]
        ]
        
        repeat i length? grad-b [
            if all [i <= length? a-data i <= length? b-data b-data/:i <> 0.0] [
                grad-b/:i: grad-b/:i * (0.0 - (a-data/:i / (b-data/:i * b-data/:i)))
            ]
        ]
        
        reduce [grad-a grad-b]
    ]
    
    ;; Power operation gradient
    pow-grad: func [a-data power-val grad-output] [
        ; d/dx (x^n) = n*x^(n-1)
        grad-a: copy grad-output
        
        repeat i length? grad-a [
            if i <= length? a-data [
                grad-a/:i: grad-a/:i * (power-val * power a-data/:i (power-val - 1.0))
            ]
        ]
        
        reduce [grad-a]
    ]
    
    ;; Sigmoid operation gradient
    sigmoid-grad: func [output-data grad-output] [
        ; d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        grad-input: copy grad-output
        
        repeat i length? grad-input [
            if i <= length? output-data [
                sig: output-data/:i
                grad-input/:i: grad-input/:i * (sig * (1.0 - sig))
            ]
        ]
        
        grad-input
    ]
    
    ;; ReLU operation gradient
    relu-grad: func [input-data grad-output] [
        ; d/dx ReLU(x) = 1 if x > 0, else 0
        grad-input: copy grad-output
        
        repeat i length? grad-input [
            if all [i <= length? input-data input-data/:i <= 0.0] [
                grad-input/:i: 0.0
            ]
        ]
        
        grad-input
    ]
    
    ;; Tanh operation gradient
    tanh-grad: func [output-data grad-output] [
        ; d/dx tanh(x) = 1 - tanh(x)^2
        grad-input: copy grad-output
        
        repeat i length? grad-input [
            if i <= length? output-data [
                tanh-val: output-data/:i
                grad-input/:i: grad-input/:i * (1.0 - (tanh-val * tanh-val))
            ]
        ]
        
        grad-input
    ]
    
    ;; Matrix multiplication gradient
    matmul-grad: func [
        "Gradient for matrix multiplication"
        a-data [block!] "First matrix data (flattened)"
        a-shape [block!] "Shape of first matrix [rows cols]"
        b-data [block!] "Second matrix data (flattened)"
        b-shape [block!] "Shape of second matrix [rows cols]"
        grad-output [block!] "Gradient from upstream"
        grad-output-shape [block!] "Shape of gradient output"
    ] [
        rows-a: a-shape/1
        cols-a: a-shape/2
        rows-b: b-shape/1
        cols-b: b-shape/2
        
        ; Gradient with respect to A is grad_output * B^T
        grad-a: make block! (rows-a * cols-a)
        loop (rows-a * cols-a) [append grad-a 0.0]
        
        repeat i rows-a [
            repeat j cols-a [
                sum: 0.0
                repeat k cols-b [
                    grad-out-idx: ((i - 1) * cols-b) + k
                    b-idx: ((k - 1) * cols-b) + j  ; Transposed index
                    if all [grad-out-idx <= length? grad-output b-idx <= length? b-data] [
                        sum: sum + (grad-output/:grad-out-idx * b-data/:b-idx)
                    ]
                ]
                a-idx: ((i - 1) * cols-a) + j
                grad-a/:a-idx: sum
            ]
        ]
        
        ; Gradient with respect to B is A^T * grad_output
        grad-b: make block! (rows-b * cols-b)
        loop (rows-b * cols-b) [append grad-b 0.0]
        
        repeat i rows-b [
            repeat j cols-b [
                sum: 0.0
                repeat k rows-a [
                    a-idx: ((k - 1) * cols-a) + i  ; Transposed index
                    grad-out-idx: ((k - 1) * cols-b) + j
                    if all [a-idx <= length? a-data grad-out-idx <= length? grad-output] [
                        sum: sum + (a-data/:a-idx * grad-output/:grad-out-idx)
                    ]
                ]
                b-idx: ((i - 1) * cols-b) + j
                grad-b/:b-idx: sum
            ]
        ]
        
        reduce [grad-a grad-b]
    ]
    
    ;; Mean operation gradient
    mean-grad: func [input-length grad-output] [
        ; Gradient is distributed evenly across all inputs
        grad-input: make block! input-length
        grad-per-input: grad-output/1 / input-length
        loop input-length [append grad-input grad-per-input]
        reduce [grad-input]
    ]
    
    ;; Sum operation gradient
    sum-grad: func [input-length grad-output] [
        ; Gradient is distributed as 1.0 to each input
        grad-input: make block! input-length
        loop input-length [append grad-input grad-output/1]
        reduce [grad-input]
    ]
    
    ;; Negation operation gradient
    neg-grad: func [grad-output] [
        neg-grad: copy grad-output
        repeat i length? neg-grad [
            neg-grad/:i: 0.0 - neg-grad/:i
        ]
        reduce [neg-grad]
    ]
]

;; Enhanced operations with gradient support
enhanced-ops: context [
    ;; Add operation with gradient tracking
    add: func [
        "Add two variables with gradient tracking"
        a [object!]
        b [object!]
    ] [
        calc-requires-grad: a/requires-grad or b/requires-grad
        
        ; Compute result
        result-data: make block! max length? a/data length? b/data
        len-a: length? a/data
        len-b: length? b/data
        max-len: max len-a len-b
        
        repeat i max-len [
            val-a: either i <= len-a [a/data/:i] [0.0]
            val-b: either i <= len-b [b/data/:i] [0.0]
            append result-data (val-a + val-b)
        ]
        
        ; Create gradient function
        local-grad-fn: func [grad-output] [
            grad-a: make block! len-a
            grad-b: make block! len-b

            repeat i len-a [
                append grad-a either i <= length? grad-output [grad-output/:i] [0.0]
            ]

            repeat i len-b [
                append grad-b either i <= length? grad-output [grad-output/:i] [0.0]
            ]

            reduce [grad-a grad-b]
        ]
        
        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: reduce [length? result-data]
            dtype: 'float32
            grad: none
            requires-grad: calc-requires-grad
            grad-fn: local-grad-fn
            parents: reduce [a b]
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
        
        ; Add as child to parents
        append a/children result-var
        append b/children result-var
        
        result-var
    ]
    
    ;; Multiply operation with gradient tracking
    mul: func [
        "Multiply two variables with gradient tracking"
        a [object!]
        b [object!]
    ] [
        requires-grad: a/requires-grad or b/requires-grad
        
        ; Compute result
        result-data: make block! max length? a/data length? b/data
        len-a: length? a/data
        len-b: length? b/data
        max-len: max len-a len-b
        
        repeat i max-len [
            val-a: either i <= len-a [a/data/:i] [1.0]
            val-b: either i <= len-b [b/data/:i] [1.0]
            append result-data (val-a * val-b)
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-a: make block! len-a
            grad-b: make block! len-b
            
            repeat i len-a [
                grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                mult-factor: either i <= len-b [b/data/:i] [1.0]
                append grad-a (grad-val * mult-factor)
            ]
            
            repeat i len-b [
                grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                mult-factor: either i <= len-a [a/data/:i] [1.0]
                append grad-b (grad-val * mult-factor)
            ]
            
            reduce [grad-a grad-b]
        ]
        
        ; Create result variable
        result-var: make object! [
            data: result-data
            shape: reduce [length? result-data]
            dtype: 'float32
            grad: none
            requires-grad: calc-requires-grad
            grad-fn: local-grad-fn
            parents: reduce [a b]
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
        
        ; Add as child to parents
        append a/children result-var
        append b/children result-var
        
        result-var
    ]
    
    ;; Sigmoid operation with gradient tracking
    sigmoid: func [
        "Apply sigmoid function with gradient tracking"
        x [object!]
    ] [
        requires-grad: x/requires-grad
        
        ; Compute sigmoid
        result-data: copy x/data
        repeat i length? result-data [
            exp-val: exp(0.0 - result-data/:i)
            result-data/:i: 1.0 / (1.0 + exp-val)
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-input: copy grad-output
            repeat i length? grad-input [
                if i <= length? result-data [
                    sig: result-data/:i
                    grad-input/:i: grad-input/:i * (sig * (1.0 - sig))
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
            requires-grad: calc-requires-grad
            grad-fn: local-grad-fn
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
    
    ;; ReLU operation with gradient tracking
    relu: func [
        "Apply ReLU function with gradient tracking"
        x [object!]
    ] [
        requires-grad: x/requires-grad
        
        ; Compute ReLU
        result-data: copy x/data
        repeat i length? result-data [
            if result-data/:i < 0.0 [
                result-data/:i: 0.0
            ]
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-input: copy grad-output
            repeat i length? grad-input [
                if all [i <= length? x/data x/data/:i <= 0.0] [
                    grad-input/:i: 0.0
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
            requires-grad: calc-requires-grad
            grad-fn: local-grad-fn
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
]