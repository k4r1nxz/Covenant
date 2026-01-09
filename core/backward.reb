REBOL [
    Title: "Covenant Backward Propagation"
    Description: "Backward propagation system for Covenant AI Framework"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Backward propagation system
backward-prop: context [
    ;; Topological sort of computational graph
    topological-sort: func [
        "Topologically sort nodes in computational graph"
        start-node [object!]
    ] [
        sorted: make block! 100
        visited: make block! 100
        
        visit: func [node] [
            if find visited node [return]  ; Already visited
            append visited node
            
            ; Visit all parents first (since we're going backwards)
            foreach parent node/parents [
                visit parent
            ]
            
            ; Then add current node
            append sorted node
        ]
        
        visit start-node
        sorted
    ]
    
    ;; Reverse topological sort (for backward pass)
    reverse-topo-sort: func [
        "Reverse topological sort for backward pass"
        start-node [object!]
    ] [
        normal-order: topological-sort start-node
        reverse normal-order
    ]
    
    ;; Perform backward pass
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
        if output-node/requires-grad [
            if not output-node/grad [
                output-node/grad: make block! length? output-node/data
                loop length? output-node/data [append output-node/grad 0.0]
            ]
            
            ; Accumulate gradients
            repeat i length? output-node/grad [
                if i <= length? grad-out [
                    output-node/grad/:i: output-node/grad/:i + grad-out/:i
                ]
            ]
        ]
        
        ; Get nodes in reverse topological order (for backward pass)
        nodes-to-process: reverse-topo-sort output-node
        
        ; Process each node
        foreach node nodes-to-process [
            ; If node has gradient function and requires gradients
            if all [node/grad-fn node/requires-grad] [
                ; Compute gradients with respect to inputs
                parent-grads: node/grad-fn node/grad
                
                ; Distribute gradients to parents
                repeat i length? node/parents [
                    parent: node/parents/:i
                    if parent/requires-grad [
                        ; Initialize parent gradient if needed
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
    
    ;; Zero gradients for all nodes in graph
    zero-grads: func [
        "Zero gradients for all nodes in computational graph"
        start-node [object!]
    ] [
        ; Collect all nodes in the graph
        all-nodes: make block! 100
        visited: make block! 100
        
        collect-nodes: func [node] [
            if find visited node [return]  ; Already visited
            append visited node
            append all-nodes node
            
            ; Visit all parents
            foreach parent node/parents [
                collect-nodes parent
            ]
        ]
        
        collect-nodes start-node
        
        ; Zero gradients for all nodes
        foreach node all-nodes [
            if node/grad [
                repeat i length? node/grad [
                    node/grad/:i: 0.0
                ]
            ]
        ]
    ]
    
    ;; Compute Jacobian-vector product for a function
    jvp: func [
        "Compute Jacobian-vector product"
        func-to-diff [function!]
        inputs [block!]
        vectors [block!]
    ] [
        ; This would implement forward-mode AD
        ; For now, we'll focus on backward-mode AD
        print "JVP not implemented in this version"
    ]
    
    ;; Compute vector-Jacobian product for a function
    vjp: func [
        "Compute vector-Jacobian product (backward mode)"
        func-to-diff [function!]
        inputs [block!]
        cotangent [block!]
    ] [
        ; This implements the backward pass for a given function
        print "VJP not implemented in this version"
    ]
]

;; Enhanced autograd operations with proper backward support
enhanced-autograd: context [
    ;; Addition operation with proper backward support
    add: func [
        "Add two variables with gradient tracking"
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
            val-a: either i <= len-a [a/data/:i] [0.0]
            val-b: either i <= len-b [b/data/:i] [0.0]
            append result-data (val-a + val-b)
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-a: make block! len-a
            grad-b: make block! len-b
            
            repeat i len-a [
                if i <= length? grad-output [
                    append grad-a grad-output/:i
                ] [
                    append grad-a 0.0
                ]
            ]
            
            repeat i len-b [
                if i <= length? grad-output [
                    append grad-b grad-output/:i
                ] [
                    append grad-b 0.0
                ]
            ]
            
            reduce [grad-a grad-b]
        ]
        
        ; Create new variable
        new-var: make object! [
            data: result-data
            shape: reduce [length? result-data]
            dtype: 'float32
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
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
        
        ; Add this variable as child to parents
        append a/children new-var
        append b/children new-var
        
        new-var
    ]
    
    ;; Multiplication operation with proper backward support
    mul: func [
        "Multiply two variables with gradient tracking"
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
            val-a: either i <= len-a [a/data/:i] [1.0]  ; Use 1.0 as identity for multiplication
            val-b: either i <= len-b [b/data/:i] [1.0]
            append result-data (val-a * val-b)
        ]
        
        ; Create gradient function
        grad-fn: func [grad-output] [
            grad-a: make block! len-a
            grad-b: make block! len-b
            
            repeat i len-a [
                if i <= length? grad-output [
                    grad-val: grad-output/:i * either i <= len-b [b/data/:i] [1.0]
                    append grad-a grad-val
                ] [
                    append grad-a 0.0
                ]
            ]
            
            repeat i len-b [
                if i <= length? grad-output [
                    grad-val: grad-output/:i * either i <= len-a [a/data/:i] [1.0]
                    append grad-b grad-val
                ] [
                    append grad-b 0.0
                ]
            ]
            
            reduce [grad-a grad-b]
        ]
        
        ; Create new variable
        new-var: make object! [
            data: result-data
            shape: reduce [length? result-data]
            dtype: 'float32
            grad: none
            requires-grad: requires-grad
            grad-fn: grad-fn
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
        
        ; Add this variable as child to parents
        append a/children new-var
        append b/children new-var
        
        new-var
    ]
]
