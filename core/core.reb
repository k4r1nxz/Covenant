REBOL [
    Title: "Covenant Core Module"
    Description: "Core tensor operations and data structures for Covenant AI Framework"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
    Rights: "BSD 2-Clause License"
]

;; Global flatten function to avoid duplication
flatten-data: func [
    "Recursively flatten nested blocks"
    data [block! number!]
] [
    if number? data [return reduce [data]]
    if empty? data [return copy []]

    result: copy []
    foreach item data [
        either block? item [
            append result flatten-data item
        ] [
            append result reduce [item]
        ]
    ]
    result
]

;; Calculate tensor shape
calculate-shape: func [
    "Calculate tensor shape from nested data"
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

;; Core tensor implementation
core: context [
    ;; Create a tensor (represented as a block in REBOL)
    tensor: func [
        "Create a tensor from data"
        input-data [block! number!] "Input data"
        /dtype "Specify data type"
        data-type [word!] "Data type (float32, int32, int64)"
    ] [
        ; Validate input
        if not any [block? input-data number? input-data] [
            throw "Input data must be a block or number"
        ]

        ; Determine data type
        tensor-dtype: either dtype [data-type] ['float32]

        ; Convert data to appropriate type
        processed-data: either number? input-data [
            either tensor-dtype = 'float32 [reduce [to-float input-data]] [
                either tensor-dtype = 'int32 [reduce [to-integer input-data]] [
                    reduce [to-integer input-data]  ; Default to int64 for other cases
                ]
            ]
        ] [
            ; Process block data
            flat-data: flatten-data input-data
            either tensor-dtype = 'float32 [
                ; Convert all elements to float
                result: copy []
                foreach val flat-data [
                    append result to-float val
                ]
                result
            ] [
                either tensor-dtype = 'int32 [
                    ; Convert all elements to integer
                    result: copy []
                    foreach val flat-data [
                        append result to-integer val
                    ]
                    result
                ] [
                    ; Default to int64
                    result: copy []
                    foreach val flat-data [
                        append result to-integer val
                    ]
                    result
                ]
            ]
        ]

        make object! [
            data: processed-data
            shape: calculate-shape input-data
            dtype: tensor-dtype
        ]
    ]

    ;; Create tensor of zeros
    zeros: func [
        "Create a tensor filled with zeros"
        input-shape [block!] "Shape of the tensor"
        /dtype "Specify data type"
        data-type [word!] "Data type (float32, int32, int64)"
    ] [
        size: 1
        foreach dim input-shape [size: size * dim]

        tensor-dtype: either dtype [data-type] ['float32]

        ; Use make to pre-allocate memory efficiently
        tensor-data: make block! size
        loop size [
            either tensor-dtype = 'float32 [
                append tensor-data 0.0
            ] [
                append tensor-data 0
            ]
        ]

        make object! [
            data: tensor-data
            shape: input-shape
            dtype: tensor-dtype
        ]
    ]

    ;; Create tensor of ones
    ones: func [
        "Create a tensor filled with ones"
        input-shape [block!] "Shape of the tensor"
        /dtype "Specify data type"
        data-type [word!] "Data type (float32, int32, int64)"
    ] [
        size: 1
        foreach dim input-shape [size: size * dim]

        tensor-dtype: either dtype [data-type] ['float32]

        ; Use make to pre-allocate memory efficiently
        tensor-data: make block! size
        loop size [
            either tensor-dtype = 'float32 [
                append tensor-data 1.0
            ] [
                append tensor-data 1
            ]
        ]

        make object! [
            data: tensor-data
            shape: input-shape
            dtype: tensor-dtype
        ]
    ]

    ;; Create tensor with random values
    rand: func [
        "Create a tensor filled with random values"
        input-shape [block!] "Shape of the tensor"
    ] [
        size: 1
        foreach dim input-shape [size: size * dim]
        ; Use make to pre-allocate memory efficiently
        tensor-data: make block! size
        loop size [append tensor-data random 1.0]
        make object! [
            data: tensor-data
            shape: input-shape
            dtype: 'float32
        ]
    ]

    ;; Create tensor with evenly spaced values (arange)
    arange: func [
        "Create a tensor with evenly spaced values"
        start [number!] "Starting value"
        end [number!] "Ending value (exclusive)"
        /step "Step size"
        step-size [number!] "Step size"
    ] [
        step-val: either step [step-size] [1.0]

        if step-val = 0 [throw "Step size cannot be zero"]

        ; Calculate size in advance to pre-allocate memory
        size: to-integer ((end - start) / step-val)
        data: make block! size
        current: start
        while [current < end] [
            append data current
            current: current + step-val
        ]

        make object! [
            data: data
            shape: reduce [length? data]
            dtype: 'float32
        ]
    ]

    ;; Create tensor with linearly spaced values (linspace)
    linspace: func [
        "Create a tensor with linearly spaced values"
        start [number!] "Starting value"
        end [number!] "Ending value"
        num [integer!] "Number of points"
    ] [
        if num <= 0 [throw "Number of points must be positive"]

        if num = 1 [
            make object! [
                data: reduce [start]
                shape: reduce [1]
                dtype: 'float32
            ]
        ] [
            ; Pre-allocate memory
            data: make block! num
            step: either num = 1 [0] [(end - start) / (num - 1)]
            repeat i num [
                append data (start + ((i - 1) * step))
            ]

            make object! [
                data: data
                shape: reduce [length? data]
                dtype: 'float32
            ]
        ]
    ]

    ;; Add two tensors - optimized version with improved memory management
    add: func [
        "Add two tensors"
        a [object!] "First tensor"
        b [object!] "Second tensor"
    ] [
        ; Broadcasting support - expand dimensions if needed
        broadcasted-a: a
        broadcasted-b: b

        if a/shape <> b/shape [
            ; Simple broadcasting: if one tensor has shape [1] and the other doesn't, broadcast
            if (a/shape = reduce [1]) and (length? b/shape > 0) [
                ; Broadcast a to match b's shape
                broadcasted-data: make block! length? b/data
                value-to-repeat: a/data/1
                loop length? b/data [append broadcasted-data value-to-repeat]

                broadcasted-a: make object! [
                    data: broadcasted-data
                    shape: b/shape
                    dtype: a/dtype
                ]
            ]
            if (b/shape = reduce [1]) and (length? a/shape > 0) [
                ; Broadcast b to match a's shape
                broadcasted-data: make block! length? a/data
                value-to-repeat: b/data/1
                loop length? a/data [append broadcasted-data value-to-repeat]

                broadcasted-b: make object! [
                    data: broadcasted-data
                    shape: a/shape
                    dtype: b/dtype
                ]
            ]

            ; Check again after broadcasting
            if broadcasted-a/shape <> broadcasted-b/shape [throw "Tensor shapes must match for addition (after broadcasting)"]
        ]

        ; Pre-allocate result data block
        result-data: make block! length? broadcasted-a/data
        len: length? broadcasted-a/data
        repeat i len [
            val-b: either i <= length? broadcasted-b/data [broadcasted-b/data/:i] [0]
            append result-data (broadcasted-a/data/:i + val-b)
        ]

        make object! [
            data: result-data
            shape: broadcasted-a/shape
            dtype: broadcasted-a/dtype
        ]
    ]

    ;; Multiply two tensors (element-wise) - optimized version with improved memory management
    mul: func [
        "Multiply two tensors element-wise"
        a [object!] "First tensor"
        b [object!] "Second tensor"
    ] [
        ; Broadcasting support - expand dimensions if needed
        broadcasted-a: a
        broadcasted-b: b

        if a/shape <> b/shape [
            ; Simple broadcasting: if one tensor has shape [1] and the other doesn't, broadcast
            if (a/shape = reduce [1]) and (length? b/shape > 0) [
                ; Broadcast a to match b's shape
                broadcasted-data: make block! length? b/data
                value-to-repeat: a/data/1
                loop length? b/data [append broadcasted-data value-to-repeat]

                broadcasted-a: make object! [
                    data: broadcasted-data
                    shape: b/shape
                    dtype: a/dtype
                ]
            ]
            if (b/shape = reduce [1]) and (length? a/shape > 0) [
                ; Broadcast b to match a's shape
                broadcasted-data: make block! length? a/data
                value-to-repeat: b/data/1
                loop length? a/data [append broadcasted-data value-to-repeat]

                broadcasted-b: make object! [
                    data: broadcasted-data
                    shape: a/shape
                    dtype: b/dtype
                ]
            ]

            ; Check again after broadcasting
            if broadcasted-a/shape <> broadcasted-b/shape [throw "Tensor shapes must match for multiplication (after broadcasting)"]
        ]

        ; Pre-allocate result data block
        result-data: make block! length? broadcasted-a/data
        len: length? broadcasted-a/data
        repeat i len [
            val-b: either i <= length? broadcasted-b/data [broadcasted-b/data/:i] [1]  ; Use 1 for multiplication identity
            append result-data (broadcasted-a/data/:i * val-b)
        ]

        make object! [
            data: result-data
            shape: broadcasted-a/shape
            dtype: broadcasted-a/dtype
        ]
    ]

    ;; Matrix multiplication (simplified for 2D) - optimized version
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

        make object! [
            data: result-data
            shape: result-shape
            dtype: a/dtype
        ]
    ]

    ;; Reshape tensor - optimized to avoid unnecessary copying
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

        ; Just return a new object with the same data reference but different shape
        ; This is more memory efficient than copying the data
        make object! [
            data: t/data
            shape: new-shape
            dtype: t/dtype
        ]
    ]

    ;; Sum tensor along axis - optimized version
    sum: func [
        "Sum tensor along specified axis"
        t [object!] "Input tensor"
        /axis "Axis to sum along (0-indexed)"
        axis-val [integer!] "Axis to sum along (0-indexed)"
    ] [
        ; If no axis specified, sum all elements
        if not axis [
            total: 0.0
            foreach val t/data [total: total + val]
            return make object! [
                data: reduce [total]
                shape: reduce [1]
                dtype: t/dtype
            ]
        ]

        if axis-val >= length? t/shape [throw "Axis out of range"]

        ; For simplicity, implementing for 2D tensors
        if (length? t/shape) = 2 [
            rows: t/shape/1
            cols: t/shape/2

            if axis-val = 0 [  ; Sum along rows (result is 1D with column sums)
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

            if axis-val = 1 [  ; Sum along columns (result is 1D with row sums)
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

    ;; Maximum value along axis - optimized version
    max: func [
        "Maximum value along specified axis"
        t [object!] "Input tensor"
        /axis "Axis to find max along (0-indexed)"
        axis-val [integer!] "Axis to find max along (0-indexed)"
    ] [
        ; If no axis specified, find max of all elements
        if not axis [
            max-val: first t/data
            foreach val next t/data [
                if val > max-val [max-val: val]
            ]
            return make object! [
                data: reduce [max-val]
                shape: reduce [1]
                dtype: t/dtype
            ]
        ]

        if axis-val >= length? t/shape [throw "Axis out of range"]

        ; For simplicity, implementing for 2D tensors
        if (length? t/shape) = 2 [
            rows: t/shape/1
            cols: t/shape/2

            if axis-val = 0 [  ; Max along rows (result is 1D with column maxes)
                result-data: make block! cols
                repeat j cols [
                    max-val: t/data/j
                    repeat i rows [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx > max-val [max-val: t/data/:idx]
                    ]
                    append result-data max-val
                ]
                return make object! [
                    data: result-data
                    shape: reduce [cols]
                    dtype: t/dtype
                ]
            ]

            if axis-val = 1 [  ; Max along columns (result is 1D with row maxes)
                result-data: make block! rows
                repeat i rows [
                    max-val: t/data/((i - 1) * cols + 1)
                    repeat j cols [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx > max-val [max-val: t/data/:idx]
                    ]
                    append result-data max-val
                ]
                return make object! [
                    data: result-data
                    shape: reduce [rows]
                    dtype: t/dtype
                ]
            ]
        ]

        throw "Max operation not implemented for this tensor shape"
    ]

    ;; Minimum value along axis - optimized version
    min: func [
        "Minimum value along specified axis"
        t [object!] "Input tensor"
        /axis "Axis to find min along (0-indexed)"
        axis-val [integer!] "Axis to find min along (0-indexed)"
    ] [
        ; If no axis specified, find min of all elements
        if not axis [
            min-val: first t/data
            foreach val next t/data [
                if val < min-val [min-val: val]
            ]
            return make object! [
                data: reduce [min-val]
                shape: reduce [1]
                dtype: t/dtype
            ]
        ]

        if axis-val >= length? t/shape [throw "Axis out of range"]

        ; For simplicity, implementing for 2D tensors
        if (length? t/shape) = 2 [
            rows: t/shape/1
            cols: t/shape/2

            if axis-val = 0 [  ; Min along rows (result is 1D with column mins)
                result-data: make block! cols
                repeat j cols [
                    min-val: t/data/j
                    repeat i rows [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx < min-val [min-val: t/data/:idx]
                    ]
                    append result-data min-val
                ]
                return make object! [
                    data: result-data
                    shape: reduce [cols]
                    dtype: t/dtype
                ]
            ]

            if axis-val = 1 [  ; Min along columns (result is 1D with row mins)
                result-data: make block! rows
                repeat i rows [
                    min-val: t/data/((i - 1) * cols + 1)
                    repeat j cols [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx < min-val [min-val: t/data/:idx]
                    ]
                    append result-data min-val
                ]
                return make object! [
                    data: result-data
                    shape: reduce [rows]
                    dtype: t/dtype
                ]
            ]
        ]

        throw "Min operation not implemented for this tensor shape"
    ]

    ;; Argmax - index of maximum value - optimized version
    argmax: func [
        "Index of maximum value along specified axis"
        t [object!] "Input tensor"
        /axis "Axis to find argmax along (0-indexed)"
        axis-val [integer!] "Axis to find argmax along (0-indexed)"
    ] [
        ; If no axis specified, find argmax of all elements
        if not axis [
            max-val: first t/data
            max-idx: 1
            idx: 1
            foreach val t/data [
                if val > max-val [
                    max-val: val
                    max-idx: idx
                ]
                idx: idx + 1
            ]
            return make object! [
                data: reduce [max-idx - 1]  ; Return 0-indexed position
                shape: reduce [1]
                dtype: 'int32
            ]
        ]

        if axis-val >= length? t/shape [throw "Axis out of range"]

        ; For simplicity, implementing for 2D tensors
        if (length? t/shape) = 2 [
            rows: t/shape/1
            cols: t/shape/2

            if axis-val = 0 [  ; Argmax along rows (result is 1D with column argmaxes)
                result-data: make block! cols
                repeat j cols [
                    max-val: t/data/j
                    max-idx: 1
                    repeat i rows [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx > max-val [
                            max-val: t/data/:idx
                            max-idx: i
                        ]
                    ]
                    append result-data (max-idx - 1)  ; 0-indexed
                ]
                return make object! [
                    data: result-data
                    shape: reduce [cols]
                    dtype: 'int32
                ]
            ]

            if axis-val = 1 [  ; Argmax along columns (result is 1D with row argmaxes)
                result-data: make block! rows
                repeat i rows [
                    max-val: t/data/((i - 1) * cols + 1)
                    max-idx: 1
                    repeat j cols [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx > max-val [
                            max-val: t/data/:idx
                            max-idx: j
                        ]
                    ]
                    append result-data (max-idx - 1)  ; 0-indexed
                ]
                return make object! [
                    data: result-data
                    shape: reduce [rows]
                    dtype: 'int32
                ]
            ]
        ]

        throw "Argmax operation not implemented for this tensor shape"
    ]

    ;; Argmin - index of minimum value - optimized version
    argmin: func [
        "Index of minimum value along specified axis"
        t [object!] "Input tensor"
        /axis "Axis to find argmin along (0-indexed)"
        axis-val [integer!] "Axis to find argmin along (0-indexed)"
    ] [
        ; If no axis specified, find argmin of all elements
        if not axis [
            min-val: first t/data
            min-idx: 1
            idx: 1
            foreach val t/data [
                if val < min-val [
                    min-val: val
                    min-idx: idx
                ]
                idx: idx + 1
            ]
            return make object! [
                data: reduce [min-idx - 1]  ; Return 0-indexed position
                shape: reduce [1]
                dtype: 'int32
            ]
        ]

        if axis-val >= length? t/shape [throw "Axis out of range"]

        ; For simplicity, implementing for 2D tensors
        if (length? t/shape) = 2 [
            rows: t/shape/1
            cols: t/shape/2

            if axis-val = 0 [  ; Argmin along rows (result is 1D with column argmins)
                result-data: make block! cols
                repeat j cols [
                    min-val: t/data/j
                    min-idx: 1
                    repeat i rows [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx < min-val [
                            min-val: t/data/:idx
                            min-idx: i
                        ]
                    ]
                    append result-data (min-idx - 1)  ; 0-indexed
                ]
                return make object! [
                    data: result-data
                    shape: reduce [cols]
                    dtype: 'int32
                ]
            ]

            if axis-val = 1 [  ; Argmin along columns (result is 1D with row argmins)
                result-data: make block! rows
                repeat i rows [
                    min-val: t/data/((i - 1) * cols + 1)
                    min-idx: 1
                    repeat j cols [
                        idx: ((i - 1) * cols) + j
                        if t/data/:idx < min-val [
                            min-val: t/data/:idx
                            min-idx: j
                        ]
                    ]
                    append result-data (min-idx - 1)  ; 0-indexed
                ]
                return make object! [
                    data: result-data
                    shape: reduce [rows]
                    dtype: 'int32
                ]
            ]
        ]

        throw "Argmin operation not implemented for this tensor shape"
    ]

    ;; Print tensor
    print-tensor: func [
        "Print tensor contents"
        t [object!] "Tensor to print"
    ] [
        print ["Tensor shape:" mold t/shape]
        print ["Tensor data:" mold t/data]
        print ["Tensor dtype:" t/dtype]
    ]

    ;; Enhanced tensor function with gradient support
    tensor-grad: func [
        "Create a tensor with optional gradient tracking"
        input-data [block! number!] "Input data"
        /requires-grad "Enable gradient tracking"
    ] [
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

    ;; Concatenate tensors along an axis - optimized version
    concat: func [
        "Concatenate tensors along an axis"
        tensors [block!] "Block of tensors to concatenate"
        /axis "Axis along which to concatenate (default: 0)"
        axis-val [integer!] "Axis value"
    ] [
        if empty? tensors [throw "Cannot concatenate empty list of tensors"]

        axis-to-use: either axis [axis-val] [0]

        ; All tensors must have the same shape except in the concatenation dimension
        base-shape: first tensors/1/shape
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
        result-shape: make block! length? base-shape
        foreach dim base-shape [append result-shape dim]
        total-size: 0
        foreach tensor tensors [
            total-size: total-size + tensor/shape/(axis-to-use + 1)
        ]
        result-shape/(axis-to-use + 1): total-size

        ; Pre-allocate result data block for efficiency
        result-data: make block! total-size
        foreach tensor tensors [
            foreach val tensor/data [
                append result-data val
            ]
        ]

        make object! [
            data: result-data
            shape: result-shape
            dtype: first tensors/1/dtype
        ]
    ]

    ;; Stack tensors along a new axis - optimized version
    stack: func [
        "Stack tensors along a new axis"
        tensors [block!] "Block of tensors to stack"
        /dim "Dimension along which to stack (default: 0)"
        dim-val [integer!] "Dimension value"
    ] [
        if empty? tensors [throw "Cannot stack empty list of tensors"]

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

        ; Pre-allocate result data block for efficiency
        total-size: length? first tensors/1/data * length? tensors
        result-data: make block! total-size
        foreach tensor tensors [
            foreach val tensor/data [
                append result-data val
            ]
        ]

        make object! [
            data: result-data
            shape: result-shape
            dtype: first tensors/1/dtype
        ]
    ]

    ;; Mean reduction - optimized version
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

    ;; Flatten tensor - optimized version
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

        ; Return a new object with the same data but different shape
        make object! [
            data: t/data
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

        ; Just return a new object with the same data reference but different shape
        make object! [
            data: t/data  ; Share the same data reference
            shape: new-shape
            dtype: t/dtype
        ]
    ]

    ;; Clone tensor - optimized version
    clone: func [
        "Clone a tensor"
        t [object!] "Input tensor"
    ] [
        ; Only copy the data if necessary, otherwise share the reference
        make object! [
            data: copy t/data
            shape: copy t/shape
            dtype: t/dtype
        ]
    ]

    ;; Transpose 2D tensor - optimized version
    transpose: func [
        "Transpose a 2D tensor"
        t [object!] "Input tensor"
    ] [
        if length? t/shape <> 2 [throw "Transpose only supports 2D tensors"]

        rows: t/shape/1
        cols: t/shape/2

        ; Pre-allocate result data
        result-data: make block! (rows * cols)
        loop (rows * cols) [append result-data 0.0]

        repeat i rows [
            repeat j cols [
                src-idx: ((i - 1) * cols) + j
                dst-idx: ((j - 1) * rows) + i
                result-data/:dst-idx: t/data/:src-idx
            ]
        ]

        make object! [
            data: result-data
            shape: reduce [cols rows]  ; Swap dimensions
            dtype: t/dtype
        ]
    ]

    ;; Power operation - optimized version
    pow: func [
        "Raise tensor to a power"
        t [object!] "Input tensor"
        exponent [number!] "Power to raise to"
    ] [
        ; Pre-allocate result data
        result-data: make block! length? t/data
        repeat i length? t/data [
            append result-data power t/data/:i exponent
        ]

        make object! [
            data: result-data
            shape: t/shape
            dtype: t/dtype
        ]
    ]

    ;; Square root - optimized version
    sqrt: func [
        "Square root of tensor"
        t [object!] "Input tensor"
    ] [
        ; Pre-allocate result data
        result-data: make block! length? t/data
        repeat i length? t/data [
            append result-data square-root t/data/:i
        ]

        make object! [
            data: result-data
            shape: t/shape
            dtype: t/dtype
        ]
    ]

    ;; Helper function for square root
    square-root: func [
        "Calculate square root using Newton's method"
        x [number!]
    ] [
        if x <= 0 [return 0.0]
        guess: x / 2.0
        repeat i 10 [  ; 10 iterations should be enough
            guess: (guess + x / guess) / 2.0
        ]
        guess
    ]

    ;; Exponential - optimized version
    exp: func [
        "Exponential of tensor"
        t [object!] "Input tensor"
    ] [
        ; Pre-allocate result data
        result-data: make block! length? t/data
        repeat i length? t/data [
            append result-data exp t/data/:i
        ]

        make object! [
            data: result-data
            shape: t/shape
            dtype: t/dtype
        ]
    ]

    ;; Natural logarithm - optimized version
    log: func [
        "Natural logarithm of tensor"
        t [object!] "Input tensor"
    ] [
        ; Pre-allocate result data
        result-data: make block! length? t/data
        repeat i length? t/data [
            append result-data log-e t/data/:i
        ]

        make object! [
            data: result-data
            shape: t/shape
            dtype: t/dtype
        ]
    ]

]
