REBOL [
    Title: "Covenant Utilities Module"
    Description: "Utility functions for Covenant AI Framework"
    Version: 1.0.0
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

    ;; Save model to file
    save-model: func [
        "Save model parameters to file"
        model [object!] "Model to save"
        file-path [file!] "Path to save file"
    ] [
        ; Convert model to a saveable format
        model-data: make block! 0
        if model/weights [
            append model-data reduce ['weights model/weights/data]
        ]
        if model/bias [
            append model-data reduce ['bias model/bias/data]
        ]

        ; Write to file
        write file-path mold model-data
        print ["Model saved to" file-path]
    ]

    ;; Load model from file
    load-model: func [
        "Load model parameters from file"
        file-path [file!] "Path to load file"
    ] [
        if not exists? file-path [
            print ["Model file does not exist:" file-path]
            return none
        ]

        model-data: load file-path
        print ["Model loaded from" file-path]
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
]