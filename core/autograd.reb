REBOL [
    Title: "Covenant Autograd System"
    Description: "Proper automatic differentiation system for Covenant AI Framework"
    Version: 1.2.0
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
        is-leaf: true  ; Initially all nodes are leaves
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
        child/is-leaf: false  ; Child is no longer a leaf
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
        node/is-leaf: false  ; Node is no longer a leaf if it has parents
    ]
]

;; Topological sort for computational graph - optimized version
topological-sort: func [
    "Topologically sort nodes in computational graph"
    nodes [block!]
] [
    ; Pre-allocate arrays for efficiency
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

;; Backpropagation function - optimized version
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

    ; Get all nodes in the computational graph - optimized collection
    all-nodes: make block! 100
    visited-nodes: make block! 100
    queue: copy reduce [output-node]

    while [not empty? queue] [
        current: take queue
        if not find visited-nodes current [
            append visited-nodes current
            append all-nodes current
            foreach parent current/parents [
                if not find queue parent [
                    append queue parent
                ]
            ]
        ]
    ]

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

                    ; Accumulate gradients efficiently
                    parent-grad-block: parent-grads/:i
                    len: min length? parent/grad length? parent-grad-block
                    repeat j len [
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

    ;; Add operation - optimized version
    add-op: func [
        "Add operation with gradient computation"
        a [object!]
        b [object!]
    ] [
        ; Check if either requires gradients
        requires-grad: a/requires-grad or b/requires-grad

        ; Pre-allocate result data
        result-data: make block! max length? a/data length? b/data
        len-a: length? a/data
        len-b: length? b/data
        max-len: max len-a len-b

        ; Compute result with broadcasting support
        repeat i max-len [
            val-a: either i <= len-a [a/data/:i] [0.0]
            val-b: either i <= len-b [b/data/:i] [0.0]
            append result-data (val-a + val-b)
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
                grad-a: make block! len-a
                repeat i len-a [
                    append grad-a either i <= length? grad-output [grad-output/:i] [0.0]
                ]

                ; Gradient with respect to b is the same as gradient output
                grad-b: make block! len-b
                repeat i len-b [
                    append grad-b either i <= length? grad-output [grad-output/:i] [0.0]
                ]

                reduce [grad-a grad-b]
            ]
        ]

        result-node
    ]

    ;; Multiply operation - optimized version
    mul-op: func [
        "Multiply operation with gradient computation"
        a [object!]
        b [object!]
    ] [
        ; Check if either requires gradients
        requires-grad: a/requires-grad or b/requires-grad

        ; Pre-allocate result data
        result-data: make block! max length? a/data length? b/data
        len-a: length? a/data
        len-b: length? b/data
        max-len: max len-a len-b

        ; Compute result with broadcasting support
        repeat i max-len [
            val-a: either i <= len-a [a/data/:i] [1.0]
            val-b: either i <= len-b [b/data/:i] [1.0]
            append result-data (val-a * val-b)
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
                grad-a: make block! len-a
                repeat i len-a [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    b-val: either i <= len-b [b/data/:i] [1.0]
                    append grad-a (grad-val * b-val)
                ]

                ; Gradient with respect to b is grad_output * a
                grad-b: make block! len-b
                repeat i len-b [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    a-val: either i <= len-a [a/data/:i] [1.0]
                    append grad-b (grad-val * a-val)
                ]

                reduce [grad-a grad-b]
            ]
        ]

        result-node
    ]

    ;; Matrix multiplication operation - optimized version
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

        if cols-a <> rows-b [throw "Matrix dimensions incompatible for multiplication"]

        ; Create result tensor
        result-shape: reduce [rows-a cols-b]
        ; Pre-allocate result data
        result-data: make block! (rows-a * cols-b)
        loop (rows-a * cols-b) [append result-data 0.0]

        ; Perform matrix multiplication with cache-friendly access pattern
        repeat i rows-a [
            repeat j cols-b [
                sum: 0.0
                repeat k cols-a [
                    idx-a: ((i - 1) * cols-a) + k
                    idx-b: ((k - 1) * cols-b) + j
                    val-a: a/data/:idx-a
                    val-b: b/data/:idx-b
                    sum: sum + (val-a * val-b)
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

    ;; Power operation with gradient computation - optimized version
    pow-op: func [
        "Power operation with gradient computation"
        base [object!]
        exponent [number!]
    ] [
        requires-grad: base/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? base/data
        ; Compute result: base^exponent
        repeat i length? base/data [
            append result-data power base/data/:i exponent
        ]

        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad

        ; Set up computational graph connections
        if requires-grad [
            add-child base result-node
            add-parent result-node base

            ; Define gradient function: d/dx(x^n) = n*x^(n-1)
            result-node/grad-fn: func [grad-output] [
                grad-base: make block! length? base/data
                exp-minus-one: exponent - 1
                repeat i length? base/data [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    base-val-pow: power base/data/:i exp-minus-one
                    append grad-base (grad-val * exponent * base-val-pow)
                ]

                reduce [grad-base]
            ]
        ]

        result-node
    ]

    ;; Exponential operation with gradient computation - optimized version
    exp-op: func [
        "Exponential operation with gradient computation"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute result: e^x
        repeat i length? x/data [
            append result-data exp x/data/:i
        ]

        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad

        ; Set up computational graph connections
        if requires-grad [
            add-child x result-node
            add-parent result-node x

            ; Define gradient function: d/dx(e^x) = e^x
            result-node/grad-fn: func [grad-output] [
                grad-x: make block! length? x/data
                repeat i length? x/data [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    exp-val: result-data/:i  ; Use precomputed exp value
                    append grad-x (grad-val * exp-val)  ; grad_output * e^x
                ]

                reduce [grad-x]
            ]
        ]

        result-node
    ]

    ;; Natural logarithm operation with gradient computation - optimized version
    log-op: func [
        "Natural logarithm operation with gradient computation"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute result: ln(x)
        repeat i length? x/data [
            append result-data log-e x/data/:i
        ]

        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad

        ; Set up computational graph connections
        if requires-grad [
            add-child x result-node
            add-parent result-node x

            ; Define gradient function: d/dx(ln(x)) = 1/x
            result-node/grad-fn: func [grad-output] [
                grad-x: make block! length? x/data
                repeat i length? x/data [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    x-val: x/data/:i
                    inv-x: either x-val = 0 [0.0] [1.0 / x-val]  ; Avoid division by zero
                    append grad-x (grad-val * inv-x)  ; grad_output * (1/x)
                ]

                reduce [grad-x]
            ]
        ]

        result-node
    ]

    ;; Sine operation with gradient computation - optimized version
    sin-op: func [
        "Sine operation with gradient computation"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute result: sin(x)
        repeat i length? x/data [
            append result-data sine x/data/:i
        ]

        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad

        ; Set up computational graph connections
        if requires-grad [
            add-child x result-node
            add-parent result-node x

            ; Define gradient function: d/dx(sin(x)) = cos(x)
            result-node/grad-fn: func [grad-output] [
                grad-x: make block! length? x/data
                repeat i length? x/data [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    cos-val: cosine x/data/:i
                    append grad-x (grad-val * cos-val)  ; grad_output * cos(x)
                ]

                reduce [grad-x]
            ]
        ]

        result-node
    ]

    ;; Cosine operation with gradient computation - optimized version
    cos-op: func [
        "Cosine operation with gradient computation"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Pre-allocate result data
        result-data: make block! length? x/data
        ; Compute result: cos(x)
        repeat i length? x/data [
            append result-data cosine x/data/:i
        ]

        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad

        ; Set up computational graph connections
        if requires-grad [
            add-child x result-node
            add-parent result-node x

            ; Define gradient function: d/dx(cos(x)) = -sin(x)
            result-node/grad-fn: func [grad-output] [
                grad-x: make block! length? x/data
                repeat i length? x/data [
                    grad-val: either i <= length? grad-output [grad-output/:i] [0.0]
                    neg-sin-val: 0.0 - sine x/data/:i  ; -sin(x)
                    append grad-x (grad-val * neg-sin-val)  ; grad_output * (-sin(x))
                ]

                reduce [grad-x]
            ]
        ]

        result-node
    ]

    ;; Mean operation with gradient computation - optimized version
    mean-op: func [
        "Mean operation with gradient computation"
        x [object!]
    ] [
        requires-grad: x/requires-grad

        ; Compute mean
        total: 0.0
        foreach val x/data [total: total + val]
        n: length? x/data
        mean-val: either n = 0 [0.0] [total / n]
        result-data: reduce [mean-val]

        ; Create result node
        result-node: graph-node/requires-grad result-data requires-grad

        ; Set up computational graph connections
        if requires-grad [
            add-child x result-node
            add-parent result-node x

            ; Define gradient function: d/dx_i(mean) = 1/n for all i
            result-node/grad-fn: func [grad-output] [
                grad-x: make block! n
                grad-scaler: either n = 0 [0.0] [grad-output/1 / n]  ; Distribute gradient equally
                loop n [append grad-x grad-scaler]

                reduce [grad-x]
            ]
        ]

        result-node
    ]

]
