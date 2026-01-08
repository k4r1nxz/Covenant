REBOL [
    Title: "Covenant Central Graph Manager"
    Description: "Centralized computational graph management for Covenant AI Framework"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Central Graph Manager - Single source of truth for computational graph
graph-manager: context [
    ;; Global registry of all nodes in the computational graph
    nodes: make block! 1000
    
    ;; Registry of leaf nodes (inputs, parameters)
    leaf-nodes: make block! 100
    
    ;; Registry of intermediate nodes (results of operations)
    intermediate-nodes: make block! 500
    
    ;; Registry of output nodes
    output-nodes: make block! 50
    
    ;; Topologically sorted nodes for forward pass
    forward-order: make block! 1000
    
    ;; Reverse topologically sorted nodes for backward pass
    backward-order: make block! 1000
    
    ;; Add a node to the graph
    add-node: func [
        "Add a node to the computational graph"
        node [object!] "Node to add"
    ] [
        if not find nodes node [
            append nodes node
            if node/is-leaf [
                append leaf-nodes node
            ] [
                append intermediate-nodes node
            ]
        ]
    ]
    
    ;; Register an output node
    register-output: func [
        "Register an output node"
        node [object!] "Output node to register"
    ] [
        if not find output-nodes node [
            append output-nodes node
        ]
    ]
    
    ;; Clear the graph
    clear-graph: func [] [
        clear nodes
        clear leaf-nodes
        clear intermediate-nodes
        clear output-nodes
        clear forward-order
        clear backward-order
    ]
    
    ;; Build topological ordering for forward pass
    build-forward-order: func [] [
        clear forward-order
        visited: make block! length? nodes
        
        visit: func [node] [
            if find visited node [return]  ; Already visited
            append visited node
            
            ; Visit all children first (dependencies)
            foreach child node/children [
                visit child
            ]
            
            ; Then add current node
            append forward-order node
        ]
        
        ; Visit all leaf nodes to start the traversal
        foreach node leaf-nodes [
            visit node
        ]
        
        ; Also visit any nodes that weren't reached through leaves
        foreach node nodes [
            if not find visited node [
                visit node
            ]
        ]
    ]
    
    ;; Build reverse topological ordering for backward pass
    build-backward-order: func [] [
        clear backward-order
        visited: make block! length? nodes
        
        visit: func [node] [
            if find visited node [return]  ; Already visited
            append visited node
            
            ; Visit all parents first (dependencies in reverse direction)
            foreach parent node/parents [
                visit parent
            ]
            
            ; Then add current node
            append backward-order node
        ]
        
        ; Visit all output nodes to start the traversal
        foreach node output-nodes [
            visit node
        ]
        
        ; Also visit any nodes that weren't reached through outputs
        foreach node nodes [
            if not find visited node [
                visit node
            ]
        ]
        
        ; Reverse the order to get proper backward flow
        reverse backward-order
    ]
    
    ;; Perform forward pass through the graph
    forward-pass: func [
        "Perform forward pass through the computational graph"
        start-nodes [block!] "Starting nodes for forward pass"
    ] [
        build-forward-order
        foreach node forward-order [
            ; Execute forward computation if needed
            ; (In this simplified version, computations are done on-demand)
        ]
    ]
    
    ;; Perform backward pass through the graph
    backward-pass: func [
        "Perform backward pass through the computational graph"
        output-node [object!] "Output node to start backward pass from"
        /grad-output "Gradient with respect to output"
        grad-out [block!]
    ] [
        ; Initialize output gradient if not provided
        if not grad-output [
            grad-out: make block! length? output-node/data
            loop length? output-node/data [append grad-out 1.0]
        ]
        
        ; Set gradient for output node
        if output-node/requires-grad [
            if not output-node/grad [
                output-node/grad: make block! length? output-node/data
                loop length? output-node/data [append output-node/grad 0.0]
            ]
            
            repeat i length? output-node/grad [
                if i <= length? grad-out [
                    output-node/grad/:i: output-node/grad/:i + grad-out/:i
                ]
            ]
        ]
        
        ; Build backward order
        build-backward-order
        
        ; Propagate gradients backwards
        foreach node backward-order [
            if all [node/grad node/grad-fn node/requires-grad] [
                ; Compute gradients with respect to inputs
                parent-grads: node/grad-fn node/grad
                
                ; Distribute gradients to parents
                repeat i length? node/parents [
                    parent: node/parents/:i
                    if parent/requires-grad [
                        if not parent/grad [
                            parent/grad: make block! length? parent/data
                            loop length? parent/data [append parent/grad 0.0]
                        ]
                        
                        ; Accumulate gradients
                        parent-grad-block: parent-grads/:i
                        repeat j length? parent/grad [
                            if j <= length? parent-grad-block [
                                parent/grad/:j: parent/grad/:j + parent-grad-block/:j
                            ]
                        ]
                    ]
                ]
            ]
        ]
    ]
    
    ;; Zero gradients for all nodes in the graph
    zero-grads: func [] [
        foreach node nodes [
            if node/grad [
                repeat i length? node/grad [
                    node/grad/:i: 0.0
                ]
            ]
        ]
    ]
    
    ;; Get all nodes that require gradients
    get-gradient-nodes: func [] [
        grad-nodes: make block! 100
        foreach node nodes [
            if node/requires-grad [
                append grad-nodes node
            ]
        ]
        grad-nodes
    ]
    
    ;; Get graph statistics
    get-stats: func [] [
        make object! [
            total-nodes: length? nodes
            leaf-nodes-count: length? leaf-nodes
            intermediate-nodes-count: length? intermediate-nodes
            output-nodes-count: length? output-nodes
            forward-order-count: length? forward-order
            backward-order-count: length? backward-order
        ]
    ]
]

;; Enhanced variable class that registers with the central graph manager
central-variable: func [
    "Create a variable that registers with the central graph manager"
    data [block! number!] "Input data"
    /requires-grad "Whether to track gradients"
    grad-flag [logic!] "Flag to track gradients"
] [
    ; Convert number to block if needed
    input-data: either number? data [reduce [data]] [copy data]
    
    ; Create the variable object
    var: make object! [
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
            if not gradient [
                grad-data: make block! length? self/data
                loop length? self/data [append grad-data 1.0]
            ]
            
            ; Use the central graph manager for backward pass
            graph-manager/backward-pass/self grad-data
        ]
    ]
    
    ; Register the variable with the central graph manager
    graph-manager/add-node var
    
    var
]

;; Enhanced operation functions that register with the central graph manager
central-ops: context [
    ;; Add operation with central graph management
    add: func [
        "Add two variables with central graph management"
        a [object!]
        b [object!]
    ] [
        ; Check if either requires gradients
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
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]
                
                ; Use the central graph manager for backward pass
                graph-manager/backward-pass/self grad-data
            ]
        ]
        
        ; Register with central graph manager
        graph-manager/add-node result-var
        
        ; Add as child to parents
        append a/children result-var
        append b/children result-var
        
        result-var
    ]
    
    ;; Multiply operation with central graph management
    mul: func [
        "Multiply two variables with central graph management"
        a [object!]
        b [object!]
    ] [
        ; Check if either requires gradients
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
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]
                
                ; Use the central graph manager for backward pass
                graph-manager/backward-pass/self grad-data
            ]
        ]
        
        ; Register with central graph manager
        graph-manager/add-node result-var
        
        ; Add as child to parents
        append a/children result-var
        append b/children result-var
        
        result-var
    ]
    
    ;; Sigmoid operation with central graph management
    sigmoid: func [
        "Apply sigmoid function with central graph management"
        x [object!]
    ] [
        calc-requires-grad: x/requires-grad
        
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
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]
                
                ; Use the central graph manager for backward pass
                graph-manager/backward-pass/self grad-data
            ]
        ]
        
        ; Register with central graph manager
        graph-manager/add-node result-var
        
        ; Add as child to parent
        append x/children result-var
        
        result-var
    ]
    
    ;; ReLU operation with central graph management
    relu: func [
        "Apply ReLU function with central graph management"
        x [object!]
    ] [
        calc-requires-grad: x/requires-grad
        
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
                if not gradient [
                    grad-data: make block! length? self/data
                    loop length? self/data [append grad-data 1.0]
                ]
                
                ; Use the central graph manager for backward pass
                graph-manager/backward-pass/self grad-data
            ]
        ]
        
        ; Register with central graph manager
        graph-manager/add-node result-var
        
        ; Add as child to parent
        append x/children result-var
        
        result-var
    ]
]

;; Export the central graph manager and operations
central-graph: graph-manager