REBOL [
    Title: "Covenant Core Module"
    Description: "Core tensor operations and data structures for Covenant AI Framework"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
    Rights: "BSD 2-Clause License"
]

;; Core tensor implementation
core: context [
    ;; Create a tensor (represented as a block in REBOL)
    tensor: func [
        "Create a tensor from data"
        input-data [block! number!] "Input data"
    ] [
        ;; Flatten nested blocks recursively
        flatten-data: func [
            data [block! number!]
        ] [
            result: copy []
            foreach item data [
                either block? item [
                    foreach subitem item [
                        either block? subitem [
                            foreach subsubitem subitem [
                                append result reduce [subsubitem]
                            ]
                        ] [
                            append result reduce [subitem]
                        ]
                    ]
                ] [
                    append result reduce [item]
                ]
            ]
            result
        ]

        make object! [
            data: either number? input-data [reduce [input-data]] [either block? input-data [flatten-data input-data] [input-data]]
            shape: calculate-shape input-data
            dtype: 'float32  ; Default type
        ]
    ]
    
    ;; Calculate tensor shape
    calculate-shape: func [
        data [block! number!]
    ] [
        ; Handle scalar case
        if number? data [return reduce [1]]

        ; Determine if data is nested (multi-dimensional)
        if empty? data [return reduce [0]]

        ; Check if first element is a block (indicating multi-dimensional data)
        if block? first data [
            outer-len: length? data
            inner-shape: calculate-shape first data
            return append reduce [outer-len] inner-shape
        ]

        ; Otherwise it's a 1D tensor
        reduce [length? data]
    ]
    
    ;; Create tensor of zeros
    zeros: func [
        "Create a tensor filled with zeros"
        input-shape [block!] "Shape of the tensor"
    ] [
        size: 1
        foreach dim input-shape [size: size * dim]
        tensor-data: make block! size
        loop size [append tensor-data 0.0]
        make object! [
            data: tensor-data
            shape: input-shape
            dtype: 'float32
        ]
    ]

    ;; Create tensor of ones
    ones: func [
        "Create a tensor filled with ones"
        input-shape [block!] "Shape of the tensor"
    ] [
        size: 1
        foreach dim input-shape [size: size * dim]
        tensor-data: make block! size
        loop size [append tensor-data 1.0]
        make object! [
            data: tensor-data
            shape: input-shape
            dtype: 'float32
        ]
    ]

    ;; Create tensor with random values
    rand: func [
        "Create a tensor filled with random values"
        input-shape [block!] "Shape of the tensor"
    ] [
        size: 1
        foreach dim input-shape [size: size * dim]
        tensor-data: make block! size
        loop size [append tensor-data random 1.0]
        make object! [
            data: tensor-data
            shape: input-shape
            dtype: 'float32
        ]
    ]
    
    ;; Add two tensors
    add: func [
        "Add two tensors"
        a [object!] "First tensor"
        b [object!] "Second tensor"
    ] [
        if a/shape <> b/shape [throw "Tensor shapes must match for addition"]
        result-data: copy a/data
        repeat i length? result-data [
            result-data/:i: result-data/:i + b/data/:i
        ]
        make object! [
            data: result-data
            shape: a/shape
            dtype: a/dtype
        ]
    ]

    ;; Multiply two tensors (element-wise)
    mul: func [
        "Multiply two tensors element-wise"
        a [object!] "First tensor"
        b [object!] "Second tensor"
    ] [
        if a/shape <> b/shape [throw "Tensor shapes must match for multiplication"]
        result-data: copy a/data
        repeat i length? result-data [
            result-data/:i: result-data/:i * b/data/:i
        ]
        make object! [
            data: result-data
            shape: a/shape
            dtype: a/dtype
        ]
    ]
    
    ;; Matrix multiplication (simplified for 2D)
    matmul: func [
        "Matrix multiplication"
        a [object!] "First tensor (matrix)"
        b [object!] "Second tensor (matrix)"
    ] [
        ; Check if both tensors are 2D
        if (length? a/shape) <> 2 [throw "First tensor must be 2D for matmul"]
        if (length? b/shape) <> 2 [throw "Second tensor must be 2D for matmul"]

        rows-a: a/shape/1
        cols-a: a/shape/2
        rows-b: b/shape/1
        cols-b: b/shape/2

        if cols-a <> rows-b [throw "Matrix dimensions incompatible for multiplication"]

        ; Create result tensor
        result-shape: reduce [rows-a cols-b]
        result: zeros result-shape

        ; Perform matrix multiplication
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
                result/data/:idx-result: sum
            ]
        ]

        result
    ]
    
    ;; Reshape tensor
    reshape: func [
        "Reshape a tensor"
        t [object!] "Input tensor"
        new-shape [block!] "New shape"
    ] [
        total-elements: 1
        foreach dim t/shape [total-elements: total-elements * dim]

        new-total: 1
        foreach dim new-shape [new-total: new-total * dim]

        if total-elements <> new-total [
            throw "Cannot reshape: element count mismatch"
        ]

        make object! [
            data: copy t/data
            shape: new-shape
            dtype: t/dtype
        ]
    ]

    ;; Sum tensor along axis
    sum: func [
        "Sum tensor along specified axis"
        t [object!] "Input tensor"
        axis [integer!] "Axis to sum along (0-indexed)"
    ] [
        if axis >= length? t/shape [throw "Axis out of range"]

        ; For simplicity, implementing for 2D tensors
        if (length? t/shape) = 2 [
            rows: t/shape/1
            cols: t/shape/2

            if axis = 0 [  ; Sum along rows (result is 1D with column sums)
                result-data: make block! cols
                repeat j cols [
                    sum-val: 0.0
                    repeat i rows [
                        idx: ((i - 1) * cols) + j
                        sum-val: sum-val + t/data/:idx
                    ]
                    append result-data sum-val
                ]
                return make object! [
                    data: result-data
                    shape: reduce [cols]
                    dtype: t/dtype
                ]
            ]

            if axis = 1 [  ; Sum along columns (result is 1D with row sums)
                result-data: make block! rows
                repeat i rows [
                    sum-val: 0.0
                    repeat j cols [
                        idx: ((i - 1) * cols) + j
                        sum-val: sum-val + t/data/:idx
                    ]
                    append result-data sum-val
                ]
                return make object! [
                    data: result-data
                    shape: reduce [rows]
                    dtype: t/dtype
                ]
            ]
        ]

        throw "Sum operation not implemented for this tensor shape"
    ]

    ;; Print tensor
    print-tensor: func [
        "Print tensor contents"
        t [object!] "Tensor to print"
    ] [
        print ["Tensor shape:" mold t/shape]
        print ["Tensor data:" mold t/data]
    ]

    ;; Enhanced tensor function with gradient support
    tensor-grad: func [
        "Create a tensor with optional gradient tracking"
        input-data [block! number!] "Input data"
        /requires-grad "Enable gradient tracking"
    ] [
        ;; Flatten nested blocks recursively
        flatten-data: func [
            data [block! number!]
        ] [
            if number? data [return reduce [data]]
            if empty? data [return copy []]

            result: copy []
            foreach item data [
                either block? item [
                    foreach subitem item [
                        either block? subitem [
                            foreach subsubitem subitem [
                                append result reduce [subsubitem]
                            ]
                        ] [
                            append result reduce [subitem]
                        ]
                    ]
                ] [
                    append result reduce [item]
                ]
            ]
            result
        ]

        base-object: make object! [
            data: either number? input-data [reduce [input-data]] [either block? input-data [flatten-data input-data] [input-data]]
            shape: calculate-shape input-data
            dtype: 'float32  ; Default type
        ]

        ;; If gradient tracking is required, extend the object
        either requires-grad [
            grad-zeros: zeros base-object/shape
            make object! [
                data: base-object/data
                shape: base-object/shape
                dtype: base-object/dtype
                grad: grad-zeros/data
                requires-grad: true
                grad-fn: none
                parents: make block! 0
            ]
        ] [
            base-object
        ]
    ]

    ;; Concatenate tensors along an axis
    concat: func [
        "Concatenate tensors along an axis"
        tensors [block!] "Block of tensors to concatenate"
        /axis "Axis along which to concatenate (default: 0)"
        axis-val [integer!] "Axis value"
    ] [
        axis-to-use: either axis [axis-val] [0]

        ; All tensors must have the same shape except in the concatenation dimension
        base-shape: tensors/1/shape
        foreach tensor tensors [
            repeat idx length? tensor/shape [
                if idx <> (axis-to-use + 1) [  ; REBOL is 1-indexed
                    if tensor/shape/:idx <> base-shape/:idx [
                        throw "Tensor shapes must match except in concatenation dimension"
                    ]
                ]
            ]
        ]

        ; Calculate result shape - create a new block
        base-shape-block: tensors/1/shape
        result-shape: make block! length? base-shape-block
        foreach dim base-shape-block [
            append result-shape dim
        ]
        total-size: 0
        foreach tensor tensors [
            total-size: total-size + tensor/shape/(axis-to-use + 1)
        ]
        result-shape/(axis-to-use + 1): total-size

        ; Concatenate data
        result-data: copy []
        foreach tensor tensors [
            append result-data copy tensor/data
        ]

        make object! [
            data: result-data
            shape: result-shape
            dtype: tensors/1/dtype
        ]
    ]

    ;; Stack tensors along a new axis
    stack: func [
        "Stack tensors along a new axis"
        tensors [block!] "Block of tensors to stack"
        /dim "Dimension along which to stack (default: 0)"
        dim-val [integer!] "Dimension value"
    ] [
        dim-to-use: either dim [dim-val] [0]

        ; All tensors must have the same shape
        base-shape: first tensors/1/shape
        foreach tensor tensors [
            if tensor/shape <> base-shape [
                throw "All tensors must have the same shape for stacking"
            ]
        ]

        ; Calculate result shape - insert number of tensors at specified dimension
        result-shape: make block! (length? base-shape + 1)
        repeat i length? base-shape [
            if i = (dim-to-use + 1) [
                append result-shape length? tensors
            ]
            append result-shape base-shape/:i
        ]

        ; Stack data
        result-data: copy []
        foreach tensor tensors [
            append result-data copy tensor/data
        ]

        make object! [
            data: result-data
            shape: result-shape
            dtype: tensors/1/dtype
        ]
    ]

    ;; Mean reduction
    mean: func [
        "Compute mean along specified axis"
        t [object!] "Input tensor"
        /axis "Axis along which to reduce"
        axis-val [integer!] "Axis value"
    ] [
        if not axis [
            ; Global mean
            total: 0.0
            foreach val t/data [total: total + val]
            mean-val: total / length? t/data
            return make object! [
                data: reduce [mean-val]
                shape: reduce [1]
                dtype: t/dtype
            ]
        ]

        ; Mean along specific axis
        result: sum t axis-val
        divisor: t/shape/(axis-val + 1)  ; Size of the reduced dimension

        result-data: copy result/data
        repeat i length? result-data [
            result-data/:i: result-data/:i / divisor
        ]

        make object! [
            data: result-data
            shape: result/shape
            dtype: t/dtype
        ]
    ]

    ;; Flatten tensor
    flatten: func [
        "Flatten tensor to 1D"
        t [object!] "Input tensor"
        /start-dim "Start dimension to flatten from"
        start-val [integer!] "Start dimension value"
        /end-dim "End dimension to flatten to"
        end-val [integer!] "End dimension value"
    ] [
        start-dim-to-use: either start-dim [start-val] [0]
        end-dim-to-use: either end-dim [end-val] [length? t/shape - 1]

        make object! [
            data: copy t/data
            shape: reduce [length? t/data]
            dtype: t/dtype
        ]
    ]

    ;; View tensor (create a view with different shape without copying data)
    view: func [
        "Create a view of tensor with different shape"
        t [object!] "Input tensor"
        new-shape [block!] "New shape"
    ] [
        total-elements: 1
        foreach dim t/shape [total-elements: total-elements * dim]

        new-total: 1
        foreach dim new-shape [new-total: new-total * dim]

        if total-elements <> new-total [
            throw "Cannot view: element count mismatch"
        ]

        make object! [
            data: copy t/data  ; Copy data to maintain independence
            shape: new-shape
            dtype: t/dtype
        ]
    ]

    ;; Clone tensor
    clone: func [
        "Clone a tensor"
        t [object!] "Input tensor"
    ] [
        make object! [
            data: copy t/data
            shape: copy t/shape
            dtype: t/dtype
        ]
    ]

]