REBOL [
    Title: "Covenant Utilities Module"
    Description: "Utility functions for Covenant AI Framework"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
    Rights: "BSD 2-Clause License"
]

;; Load data loading utilities
do %data.reb

;; Utility functions
utils: context [
    ;; Load data from file
    load-data: func [
        "Load data from a file"
        file-path [file!] "Path to data file"
        /as-dataset "Return as dataset object"
    ] [
        if as-dataset [
            data/dataset file-path
        ] [
            ; For simplicity, this would load a simple format
            ; In a real implementation, this would handle various data formats
            data: read/lines file-path
            parsed-data: make block! length? data
            foreach line data [
                append parsed-data load line
            ]
            parsed-data
        ]
    ]

    ;; Save model to file with improved serialization
    save-model: func [
        "Save model parameters to a file"
        model [object!] "Model to save"
        file-path [file!] "Path to save file"
    ] [
        ; Create a serializable representation of the model
        model-data: make object! [
            type: model/type
            timestamp: now
            params: make block! 10
        ]

        ; Extract parameters from model components
        if model/weights [
            append model-data/params make object! [
                name: 'weights
                data: copy model/weights/data
                shape: copy model/weights/shape
                requires_grad: model/weights/requires-grad
            ]
        ]

        if model/bias [
            append model-data/params make object! [
                name: 'bias
                data: copy model/bias/data
                shape: copy model/bias/shape
                requires_grad: model/bias/requires-grad
            ]
        ]

        ; Handle sequential models
        if model/layers [
            model-data/layers: make block! length? model/layers
            foreach layer model/layers [
                layer-data: make object! [
                    type: layer/type
                    params: make block! 5
                ]

                ; Extract layer parameters
                if layer/weights [
                    append layer-data/params make object! [
                        name: 'weights
                        data: copy layer/weights/data
                        shape: copy layer/weights/shape
                        requires_grad: layer/weights/requires-grad
                    ]
                ]

                if layer/bias [
                    append layer-data/params make object! [
                        name: 'bias
                        data: copy layer/bias/data
                        shape: copy layer/bias/shape
                        requires_grad: layer/bias/requires-grad
                    ]
                ]

                append model-data/layers layer-data
            ]
        ]

        ; Write to file
        write file-path mold model-data
        print ["Model saved to:" file-path "at" now/time]
    ]

    ;; Load model from file with improved deserialization
    load-model: func [
        "Load model parameters from a file"
        file-path [file!] "Path to model file"
    ] [
        if not exists? file-path [
            print ["Error: File does not exist:" file-path]
            return none
        ]

        ; Read from file
        model-data: load file-path
        print ["Model loaded from:" file-path "at" now/time]

        ; Reconstruct model objects
        if model-data/layers [
            ; Reconstruct sequential model
            layers: make block! length? model-data/layers
            foreach layer-data model-data/layers [
                layer: make object! [
                    type: layer-data/type
                ]

                foreach param layer-data/params [
                    if param/name = 'weights [
                        layer/weights: make object! [
                            data: copy param/data
                            shape: copy param/shape
                            dtype: 'float32
                            grad: none
                            requires-grad: param/requires_grad
                            grad-fn: none
                            parents: make block! 0
                            children: make block! 0
                            is-leaf: true
                        ]
                    ]

                    if param/name = 'bias [
                        layer/bias: make object! [
                            data: copy param/data
                            shape: copy param/shape
                            dtype: 'float32
                            grad: none
                            requires-grad: param/requires_grad
                            grad-fn: none
                            parents: make block! 0
                            children: make block! 0
                            is-leaf: true
                        ]
                    ]
                ]

                append layers layer
            ]

            model-data/layers: layers
        ]

        ; Reconstruct individual layer parameters
        if model-data/params [
            foreach param model-data/params [
                if param/name = 'weights [
                    model-data/weights: make object! [
                        data: copy param/data
                        shape: copy param/shape
                        dtype: 'float32
                        grad: none
                        requires-grad: param/requires_grad
                        grad-fn: none
                        parents: make block! 0
                        children: make block! 0
                        is-leaf: true
                    ]
                ]

                if param/name = 'bias [
                    model-data/bias: make object! [
                        data: copy param/data
                        shape: copy param/shape
                        dtype: 'float32
                        grad: none
                        requires-grad: param/requires_grad
                        grad-fn: none
                        parents: make block! 0
                        children: make block! 0
                        is-leaf: true
                    ]
                ]
            ]
        ]

        model-data
    ]

    ;; Visualize tensor (simple text-based visualization)
    visualize: func [
        "Visualize tensor data"
        tensor [object!] "Tensor to visualize"
    ] [
        print ["Visualizing tensor with shape:" mold tensor/shape]
        print ["Data:" mold tensor/data]

        ; For 2D tensors, show in matrix format
        if (length? tensor/shape) = 2 [
            rows: tensor/shape/1
            cols: tensor/shape/2
            print "Matrix format:"
            repeat i rows [
                row-data: copy []
                repeat j cols [
                    idx: ((i - 1) * cols) + j
                    append row-data tensor/data/:idx
                ]
                print [mold row-data]
            ]
        ]
    ]

    ;; Simple training loop helper
    train-step: func [
        "Perform a single training step"
        model [object!] "Model to train"
        input [object!] "Input data"
        target [object!] "Target data"
        loss-fn [word!] "Loss function to use"
        optimizer [object!] "Optimizer"
    ] [
        ; Forward pass
        prediction: model/forward input

        ; Calculate loss
        loss: do reduce [loss-fn prediction target]

        ; For this simplified version, we'll skip backpropagation
        ; In a real implementation, this would compute gradients

        ; Return loss
        loss
    ]

    ;; Evaluate model performance
    evaluate: func [
        "Evaluate model performance"
        model [object!] "Model to evaluate"
        test-data [block!] "Test data"
        loss-fn [word!] "Loss function to use"
    ] [
        total-loss: 0.0
        count: 0

        foreach item test-data [
            input: item/1
            target: item/2
            prediction: model/forward input
            loss: do reduce [loss-fn prediction target]
            total-loss: total-loss + loss
            count: count + 1
        ]

        total-loss / count
    ]

    ;; Serialize any object to string
    serialize: func [
        "Serialize an object to a string representation"
        obj [any-type!] "Object to serialize"
    ] [
        ; Convert object to string representation
        mold obj
    ]

    ;; Deserialize string to object
    deserialize: func [
        "Deserialize a string representation back to an object"
        str [string!] "String representation to deserialize"
    ] [
        ; Convert string back to object
        load str
    ]

    ;; Pretty print tensor
    pprint: func [
        "Pretty print a tensor with formatting"
        tensor [object!] "Tensor to print"
        /title "Title for the print"
        title-str [string!] "Title string"
    ] [
        if title [print ["^n" title-str ":"]]
        print ["Shape:" mold tensor/shape]
        print ["DType:" tensor/dtype]

        ; Format data based on size
        if length? tensor/data <= 10 [
            print ["Data:" mold tensor/data]
        ] [
            print ["Data (first 10):" mold copy/part tensor/data 10]
            print ["... and" (length? tensor/data - 10) "more elements"]
        ]
    ]

    ;; Check if a tensor contains NaN or infinity
    validate-tensor: func [
        "Check if a tensor contains NaN or infinity values"
        tensor [object!] "Tensor to validate"
    ] [
        has-nan: false
        has-inf: false

        foreach val tensor/data [
            if val <> val [has-nan: true]  ; NaN check: NaN is not equal to itself
            if val > 1e308 [has-inf: true]  ; Infinity check: arbitrary large number
        ]

        make object! [
            is-valid: not any [has-nan has-inf]
            has_nan: has-nan
            has_inf: has-inf
            message: either not is-valid [
                join "" [
                    either has-nan ["NaN detected; "] [""]
                    either has-inf ["Infinity detected; "] [""]
                ]
            ] ["Valid tensor"]
        ]
    ]
]
