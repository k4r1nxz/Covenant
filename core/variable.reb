REBOL [
    Title: "Covenant Variable Class"
    Description: "Variable class with gradient tracking for Covenant AI Framework"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Import autograd system
do %autograd.reb

;; Variable class
variable: func [
    "Create a variable with gradient tracking"
    data [block! number!] "Input data"
    /requires-grad "Whether to track gradients"
    grad-flag [logic!] "Flag to track gradients"
] [
    ; Convert number to block if needed
    input-data: either number? data [reduce [data]] [copy data]

    ; Create the variable object
    make object! [
        data: input-data
        shape: reduce [length? input-data]
        dtype: 'float32
        grad: none
        requires-grad: either grad-flag [grad-flag] [false]
        grad-fn: none
        parents: make block! 0
        children: make block! 0
        is-leaf: true
        
        ; Method to set requires-grad
        set-requires-grad: func [flag [logic!]] [
            self/requires-grad: flag
            if flag and not self/grad [
                self/grad: make block! length? self/data
                loop length? self/data [append self/grad 0.0]
            ]
        ]
        
        ; Method to zero gradients
        zero-grad: func [] [
            if self/grad [
                repeat i length? self/grad [
                    self/grad/:i: 0.0
                ]
            ]
        ]
        
        ; Method to backward propagate gradients
        backward: func [/gradient grad-data [block!]] [
            ; Initialize gradient if not provided
            if not gradient [
                grad-data: make block! length? self/data
                loop length? self/data [append grad-data 1.0]
            ]
            
            ; Set the gradient for this variable
            if self/requires-grad [
                if not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
                
                ; Accumulate gradients
                repeat i length? self/grad [
                    if i <= length? grad-data [
                        self/grad/:i: self/grad/:i + grad-data/:i
                    ]
                ]
            ]
            
            ; Propagate to parents if this has a grad function
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

;; Function to create a new variable from operation
new-variable: func [
    "Create a new variable from an operation"
    data [block!]
    /with-parents parent-vars [block!]
    /with-grad-fn grad-func [function!]
    /requires-grad flag [logic!]
] [
    input-data: copy data
    var: make object! [
        data: input-data
        shape: reduce [length? input-data]
        dtype: 'float32
        grad: none
        requires-grad: either flag [flag] [false]
        grad-fn: either with-grad-fn [grad-func] [none]
        parents: either with-parents [copy parent-vars] [make block! 0]
        children: make block! 0
        is-leaf: false
        
        ; Method to set requires-grad
        set-requires-grad: func [flag [logic!]] [
            self/requires-grad: flag
            if flag and not self/grad [
                self/grad: make block! length? self/data
                loop length? self/data [append self/grad 0.0]
            ]
        ]
        
        ; Method to zero gradients
        zero-grad: func [] [
            if self/grad [
                repeat i length? self/grad [
                    self/grad/:i: 0.0
                ]
            ]
        ]
        
        ; Method to backward propagate gradients
        backward: func [/gradient grad-data [block!]] [
            ; Initialize gradient if not provided
            if not gradient [
                grad-data: make block! length? self/data
                loop length? self/data [append grad-data 1.0]
            ]
            
            ; Set the gradient for this variable
            if self/requires-grad [
                if not self/grad [
                    self/grad: make block! length? self/data
                    loop length? self/data [append self/grad 0.0]
                ]
                
                ; Accumulate gradients
                repeat i length? self/grad [
                    if i <= length? grad-data [
                        self/grad/:i: self/grad/:i + grad-data/:i
                    ]
                ]
            ]
            
            ; Propagate to parents if this has a grad function
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
    
    ; Add this variable as child to parents
    if with-parents [
        foreach parent parent-vars [
            if parent/children [
                append parent/children var
            ]
        ]
    ]
    
    var
]

;; Autograd operations
autograd-ops: context [
    ;; Addition operation
    add: func [
        "Add two variables"
        a [object!]
        b [object!]
    ] [
        ; Check if either requires gradients
        requires-grad: a/requires-grad or b/requires-grad
        
        ; Compute result
        result-data: copy a/data
        repeat i length? result-data [
            if i <= length? b/data [
                result-data/:i: result-data/:i + b/data/:i
            ]
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-a: copy grad-output
            grad-b: copy grad-output
            
            ; Truncate if necessary
            while [length? grad-a > length? a/data] [take/last grad-a]
            while [length? grad-b > length? b/data] [take/last grad-b]
            
            reduce [grad-a grad-b]
        ]
        
        ; Create new variable
        new-variable/with-parents reduce [a b] /with-grad-fn grad-fn /requires-grad requires-grad result-data
    ]
    
    ;; Multiplication operation
    mul: func [
        "Multiply two variables"
        a [object!]
        b [object!]
    ] [
        ; Check if either requires gradients
        requires-grad: a/requires-grad or b/requires-grad
        
        ; Compute result
        result-data: copy a/data
        repeat i length? result-data [
            if i <= length? b/data [
                result-data/:i: result-data/:i * b/data/:i
            ]
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-a: copy grad-output
            grad-b: copy grad-output
            
            ; Compute gradients: d/dx (x*y) = y and d/dy (x*y) = x
            repeat i length? grad-a [
                if i <= length? b/data [
                    grad-a/:i: grad-a/:i * b/data/:i
                ]
            ]
            repeat i length? grad-b [
                if i <= length? a/data [
                    grad-b/:i: grad-b/:i * a/data/:i
                ]
            ]
            
            ; Truncate if necessary
            while [length? grad-a > length? a/data] [take/last grad-a]
            while [length? grad-b > length? b/data] [take/last grad-b]
            
            reduce [grad-a grad-b]
        ]
        
        ; Create new variable
        new-variable/with-parents reduce [a b] /with-grad-fn grad-fn /requires-grad requires-grad result-data
    ]
    
    ;; Matrix multiplication operation
    matmul: func [
        "Matrix multiplication of two variables"
        a [object!]
        b [object!]
    ] [
        ; Check if either requires gradients
        requires-grad: a/requires-grad or b/requires-grad
        
        ; Compute matrix multiplication
        ; Assuming a is [rows_a x cols_a] and b is [cols_a x cols_b]
        ; Result will be [rows_a x cols_b]
        rows-a: a/shape/1
        cols-a: a/shape/2
        rows-b: b/shape/1
        cols-b: b/shape/2
        
        result-data: make block! (rows-a * cols-b)
        loop (rows-a * cols-b) [append result-data 0.0]
        
        repeat i rows-a [
            repeat j cols-b [
                sum: 0.0
                repeat k cols-a [
                    idx-a: ((i - 1) * cols-a) + k
                    idx-b: ((k - 1) * cols-b) + j
                    sum: sum + (a/data/:idx-a * b/data/:idx-b)
                ]
                idx-result: ((i - 1) * cols-b) + j
                result-data/:idx-result: sum
            ]
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            ; Gradient with respect to a is grad_output * b^T
            grad-a: make block! length? a/data
            loop length? a/data [append grad-a 0.0]
            
            repeat i rows-a [
                repeat j cols-a [
                    sum: 0.0
                    repeat k cols-b [
                        grad-idx: ((i - 1) * cols-b) + k
                        b-idx: ((j - 1) * cols-b) + k
                        sum: sum + (grad-output/:grad-idx * b/data/:b-idx)
                    ]
                    a-idx: ((i - 1) * cols-a) + j
                    grad-a/:a-idx: sum
                ]
            ]
            
            ; Gradient with respect to b is a^T * grad_output
            grad-b: make block! length? b/data
            loop length? b/data [append grad-b 0.0]
            
            repeat i rows-b [
                repeat j cols-b [
                    sum: 0.0
                    repeat k rows-a [
                        a-idx: ((k - 1) * cols-a) + i
                        grad-idx: ((k - 1) * cols-b) + j
                        sum: sum + (a/data/:a-idx * grad-output/:grad-idx)
                    ]
                    b-idx: ((i - 1) * cols-b) + j
                    grad-b/:b-idx: sum
                ]
            ]
            
            reduce [grad-a grad-b]
        ]
        
        ; Create new variable with shape [rows_a cols_b]
        result-var: new-variable/with-parents reduce [a b] /with-grad-fn grad-fn /requires-grad requires-grad result-data
        result-var/shape: reduce [rows-a cols-b]
        result-var
    ]
    
    ;; Mean operation
    mean: func [
        "Compute mean of a variable"
        a [object!]
    ] [
        n: length? a/data
        if n = 0 [n: 1]  ; Prevent division by zero
        
        ; Compute mean
        sum: 0.0
        foreach val a/data [sum: sum + val]
        mean-val: sum / n
        result-data: reduce [mean-val]
        
        ; Check if requires gradients
        requires-grad: a/requires-grad
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-a: make block! length? a/data
            grad-mean: first grad-output
            grad-each: grad-mean / n
            loop length? a/data [append grad-a grad-each]
            reduce [grad-a]
        ]
        
        ; Create new variable
        new-variable/with-parents reduce [a] /with-grad-fn grad-fn /requires-grad requires-grad result-data
    ]
]