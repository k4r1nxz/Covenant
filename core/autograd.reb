REBOL [
    Title: "Covenant Autograd System"
    Description: "Proper automatic differentiation system for Covenant AI Framework"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Computational graph node
graph-node: func [
    "Create a computational graph node"
    data [block!] "Tensor data"
    /requires-grad "Whether to track gradients"
    /children "Child nodes"
    child-nodes [block!]
    /operation "Operation that created this node"
    op-func [word! function!]
    /parents "Parent nodes"
    parent-nodes [block!]
    /grad-fn "Function to compute gradients"
    grad-func [function!]
] [
    make object! [
        data: data
        shape: either empty? data [reduce [0]] [reduce [length? data]]
        dtype: 'float32
        grad: either requires-grad [make block! length? data] [none]
        requires-grad: either requires-grad [true] [false]
        children: either children [child-nodes] [make block! 0]
        operation: either operation [op-func] [none]
        parents: either parents [parent-nodes] [make block! 0]
        grad-fn: either grad-fn [grad-func] [none]
        grad-computed: false
    ]
]

;; Add child to node
add-child: func [
    "Add a child node"
    node [object!]
    child [object!]
] [
    if not find node/children child [
        append node/children child
    ]
]

;; Add parent to node
add-parent: func [
    "Add a parent node"
    node [object!]
    parent [object!]
] [
    if not find node/parents parent [
        append node/parents parent
    ]
]

;; Topological sort for computational graph
topological-sort: func [
    "Topologically sort nodes in computational graph"
    nodes [block!]
] [
    sorted: make block! length? nodes
    visited: make block! length? nodes
    
    visit: func [node] [
        if find visited node [return]  ; Already visited
        append visited node
        
        ; Visit all children first
        foreach child node/children [
            visit child
        ]
        
        ; Then add current node
        append sorted node
    ]
    
    ; Visit all nodes
    foreach node nodes [
        visit node
    ]
    
    sorted
]

;; Backpropagation function
backward: func [
    "Perform backward pass to compute gradients"
    output-node [object!]
    /grad-output "Gradient with respect to output"
    grad-out [block!]
] [
    ; Initialize output gradient if not provided
    if not grad-output [
        grad-out: make block! length? output-node/data
        loop length? output-node/data [
            append grad-out 1.0
        ]
    ]
    
    ; Set output node gradient
    output-node/grad: copy grad-out
    output-node/grad-computed: true
    
    ; Get all nodes in the computational graph
    all-nodes: make block! 100
    collect-nodes: func [node] [
        if not find all-nodes node [
            append all-nodes node
            foreach parent node/parents [
                collect-nodes parent
            ]
        ]
    ]
    collect-nodes output-node
    
    ; Topologically sort nodes (from output to input)
    sorted-nodes: reverse topological-sort all-nodes
    
    ; Propagate gradients backwards
    foreach node sorted-nodes [
        if node/grad-computed and node/grad-fn [
            ; Compute gradients with respect to inputs
            parent-grads: node/grad-fn node/grad
            
            ; Distribute gradients to parents
            repeat i length? node/parents [
                parent: node/parents/:i
                if parent/requires-grad [
                    if parent/grad = none [
                        parent/grad: make block! length? parent/data
                        loop length? parent/data [append parent/grad 0.0]
                    ]
                    
                    ; Accumulate gradients
                    parent-grad-block: parent-grads/:i
                    repeat j length? parent/grad [
                        parent/grad/:j: parent/grad/:j + parent-grad-block/:j
                    ]
                    parent/grad-computed: true
                ]
            ]
        ]
    ]
]

;; Autograd module
autograd: context [
    ;; Create a variable that tracks gradients
    variable: func [
        "Create a variable that tracks gradients"
        data [block! number!] "Input data"
        /requires-grad "Whether to track gradients"
    ] [
        ; Convert number to block if needed
        if number? data [data: reduce [data]]
        
        graph-node/requires-grad data requires-grad
    ]
    
    ;; Add operation
    add-op: func [
        "Add operation with gradient computation"
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
        
        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad
        
        ; Set up computational graph connections
        if requires-grad [
            add-child a result-node
            add-child b result-node
            add-parent result-node a
            add-parent result-node b
            
            ; Define gradient function
            result-node/grad-fn: func [grad-output] [
                ; Gradient with respect to a is the same as gradient output
                grad-a: copy grad-output
                while [length? grad-a > length? a/data] [take/last grad-a]
                
                ; Gradient with respect to b is the same as gradient output
                grad-b: copy grad-output
                while [length? grad-b > length? b/data] [take/last grad-b]
                
                reduce [grad-a grad-b]
            ]
        ]
        
        result-node
    ]
    
    ;; Multiply operation
    mul-op: func [
        "Multiply operation with gradient computation"
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
        
        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad
        
        ; Set up computational graph connections
        if requires-grad [
            add-child a result-node
            add-child b result-node
            add-parent result-node a
            add-parent result-node b
            
            ; Define gradient function
            result-node/grad-fn: func [grad-output] [
                ; Gradient with respect to a is grad_output * b
                grad-a: copy grad-output
                repeat i length? grad-a [
                    if i <= length? b/data [
                        grad-a/:i: grad-a/:i * b/data/:i
                    ]
                ]
                while [length? grad-a > length? a/data] [take/last grad-a]
                
                ; Gradient with respect to b is grad_output * a
                grad-b: copy grad-output
                repeat i length? grad-b [
                    if i <= length? a/data [
                        grad-b/:i: grad-b/:i * a/data/:i
                    ]
                ]
                while [length? grad-b > length? b/data] [take/last grad-b]
                
                reduce [grad-a grad-b]
            ]
        ]
        
        result-node
    ]
    
    ;; Matrix multiplication operation
    matmul-op: func [
        "Matrix multiplication operation with gradient computation"
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
        
        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad
        result-node/shape: reduce [rows-a cols-b]
        
        ; Set up computational graph connections
        if requires-grad [
            add-child a result-node
            add-child b result-node
            add-parent result-node a
            add-parent result-node b
            
            ; Define gradient function
            result-node/grad-fn: func [grad-output] [
                ; Gradient with respect to a is grad_output * b^T
                grad-a: make block! length? a/data
                loop length? a/data [append grad-a 0.0]
                
                repeat i rows-a [
                    repeat j cols-a [
                        sum: 0.0
                        repeat k cols-b [
                            grad-idx: ((i - 1) * cols-b) + k
                            b-idx: ((k - 1) * cols-b) + j
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
        ]
        
        result-node
    ]
]