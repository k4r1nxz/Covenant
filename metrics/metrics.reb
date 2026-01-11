REBOL [
    Title: "Covenant Metrics Module"
    Description: "Evaluation metrics for Covenant AI Framework"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Import core modules
do %../core/core.reb

;; Metrics module for Covenant
metrics: context [
    ;; Calculate accuracy
    accuracy: func [
        "Calculate accuracy between predictions and targets"
        predictions [object!] "Predicted values"
        targets [object!] "Target values"
    ] [
        if length? predictions/data <> length? targets/data [
            throw "Predictions and targets must have the same length"
        ]

        correct: 0
        total: length? predictions/data

        repeat i total [
            if (round/to predictions/data/:i 0.5) = (round/to targets/data/:i 0.5) [
                correct: correct + 1
            ]
        ]

        accuracy-value: correct / total
        make object! [
            value: accuracy-value
            percentage: accuracy-value * 100
        ]
    ]

    ;; Calculate precision
    precision: func [
        "Calculate precision for binary classification"
        predictions [object!] "Predicted values (0 or 1)"
        targets [object!] "Target values (0 or 1)"
    ] [
        if length? predictions/data <> length? targets/data [
            throw "Predictions and targets must have the same length"
        ]

        tp: 0  ; True positives
        fp: 0  ; False positives

        repeat i length? predictions/data [
            pred: round/to predictions/data/:i 0.5
            target: round/to targets/data/:i 0.5
            
            if all [pred = 1 target = 1] [tp: tp + 1]
            if all [pred = 1 target = 0] [fp: fp + 1]
        ]

        precision-value: either (tp + fp) = 0 [0] [tp / (tp + fp)]
        make object! [
            value: precision-value
            percentage: precision-value * 100
        ]
    ]

    ;; Calculate recall
    recall: func [
        "Calculate recall for binary classification"
        predictions [object!] "Predicted values (0 or 1)"
        targets [object!] "Target values (0 or 1)"
    ] [
        if length? predictions/data <> length? targets/data [
            throw "Predictions and targets must have the same length"
        ]

        tp: 0  ; True positives
        fn: 0  ; False negatives

        repeat i length? predictions/data [
            pred: round/to predictions/data/:i 0.5
            target: round/to targets/data/:i 0.5
            
            if all [pred = 1 target = 1] [tp: tp + 1]
            if all [pred = 0 target = 1] [fn: fn + 1]
        ]

        recall-value: either (tp + fn) = 0 [0] [tp / (tp + fn)]
        make object! [
            value: recall-value
            percentage: recall-value * 100
        ]
    ]

    ;; Calculate F1 score
    f1-score: func [
        "Calculate F1 score for binary classification"
        predictions [object!] "Predicted values (0 or 1)"
        targets [object!] "Target values (0 or 1)"
    ] [
        prec: precision predictions targets
        rec: recall predictions targets
        
        f1-value: either (prec/value + rec/value) = 0 [0] [2 * (prec/value * rec/value) / (prec/value + rec/value)]
        make object! [
            value: f1-value
            percentage: f1-value * 100
        ]
    ]

    ;; Calculate Mean Absolute Error
    mae: func [
        "Calculate Mean Absolute Error"
        predictions [object!] "Predicted values"
        targets [object!] "Target values"
    ] [
        if length? predictions/data <> length? targets/data [
            throw "Predictions and targets must have the same length"
        ]

        total-error: 0.0
        repeat i length? predictions/data [
            total-error: total-error + abs(predictions/data/:i - targets/data/:i)
        ]

        mae-value: total-error / length? predictions/data
        make object! [
            value: mae-value
        ]
    ]

    ;; Calculate Root Mean Square Error
    rmse: func [
        "Calculate Root Mean Square Error"
        predictions [object!] "Predicted values"
        targets [object!] "Target values"
    ] [
        if length? predictions/data <> length? targets/data [
            throw "Predictions and targets must have the same length"
        ]

        total-squared-error: 0.0
        repeat i length? predictions/data [
            diff: predictions/data/:i - targets/data/:i
            total-squared-error: total-squared-error + (diff * diff)
        ]

        mse-value: total-squared-error / length? predictions/data
        rmse-value: core/square-root mse-value
        make object! [
            value: rmse-value
        ]
    ]

    ;; Calculate R-squared (coefficient of determination)
    r-squared: func [
        "Calculate R-squared (coefficient of determination)"
        predictions [object!] "Predicted values"
        targets [object!] "Target values"
    ] [
        if length? predictions/data <> length? targets/data [
            throw "Predictions and targets must have the same length"
        ]

        ; Calculate mean of targets
        sum-targets: 0.0
        foreach val targets/data [sum-targets: sum-targets + val]
        mean-targets: sum-targets / length? targets/data

        ; Calculate total sum of squares
        ss-total: 0.0
        foreach val targets/data [
            diff: val - mean-targets
            ss-total: ss-total + (diff * diff)
        ]

        ; Calculate residual sum of squares
        ss-residual: 0.0
        repeat i length? predictions/data [
            diff: targets/data/:i - predictions/data/:i
            ss-residual: ss-residual + (diff * diff)
        ]

        r-squared-value: either ss-total = 0 [1] [1 - (ss-residual / ss-total)]
        make object! [
            value: r-squared-value
        ]
    ]

    ;; Confusion matrix for binary classification
    confusion-matrix: func [
        "Calculate confusion matrix for binary classification"
        predictions [object!] "Predicted values (0 or 1)"
        targets [object!] "Target values (0 or 1)"
    ] [
        if length? predictions/data <> length? targets/data [
            throw "Predictions and targets must have the same length"
        ]

        tn: 0  ; True negatives
        fp: 0  ; False positives
        fn: 0  ; False negatives
        tp: 0  ; True positives

        repeat i length? predictions/data [
            pred: round/to predictions/data/:i 0.5
            target: round/to targets/data/:i 0.5
            
            if all [pred = 0 target = 0] [tn: tn + 1]
            if all [pred = 0 target = 1] [fn: fn + 1]
            if all [pred = 1 target = 0] [fp: fp + 1]
            if all [pred = 1 target = 1] [tp: tp + 1]
        ]

        make object! [
            tn: tn  ; True Negatives
            fp: fp  ; False Positives
            fn: fn  ; False Negatives
            tp: tp  ; True Positives
            matrix: reduce [tn fp fn tp]
        ]
    ]
]