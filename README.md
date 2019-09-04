# wavenet-finance

To configure these parameters, edit in args.json:
`nano args.json`

It is currently configured as:
```
{
    "test_round": 50,
    "batch_size": 200,
    "epochs": 20,
    "run_example": [0,1,2,3,4,5]
}
```

"test_round": denotes how many points to predict by running the iterative training/testing phase.

"run_example": denotes corresponding to-run experiments. Feel free to remove the experiments that doesn't interest you.

* "run_example" = 0 :
    experiment with outputs encoded in equal **bin_width** fashion, use_residual = True, use_skip = True, use_condition = False
* "run_example" = 1 :
    experiment with outputs encoded in equal bin_width fashion, use_residual = True, use_skip = True, use_condition = True
* "run_example" = 2 :
    experiment with outputs encoded in equal bin_width fashion, use_residual = False, use_skip = True, use_condition = False
* "run_example" = 3 :
    experiment with outputs encoded in equal bin_width fashion, use_residual = True, use_skip = False, use_condition = False
* "run_example" = 4 :
    experiment with outputs encoded in equal **bin_count** fashion, use_residual = True, use_skip = True, use_condition = False
* "run_example" = 5 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True
    
* "run_example" = 6 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True, iterative_step_train
* "run_example" = 7 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True,
    preprocess X with moving_average
* "run_example" = 8 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True,
    iterative_step_train, preprocess X with moving_average
* "run_example" = 9 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True, iterative_step_train, higher receptive field
* "run_example" = 10 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = False, iterative_step_train
* "run_example" = 11 :
    experiment with outputs encoded in equal **bin_width** fashion, use_residual = True, use_skip = True, use_condition = False, iterative_step_train
* "run_example" = 12 :
    experiment with outputs encoded in equal bin_width fashion, use_residual = True, use_skip = True, use_condition = True, iterative_step_train
* "run_example" = 13 :
    experiment with outputs encoded in equal **bin_count**  fashion, use_residual = True, use_skip = False, use_condition = True, iterative_step_train
* "run_example" = 14 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = False, use_skip = True, use_condition = True, iterative_step_train
* "run_example" = 15 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = False, use_skip = False, use_condition = True, iterative_step_train
* "run_example" = 16 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = False, use_skip = False, use_condition = False, iterative_step_train
* "run_example" = 17 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True, iterative_step_train, GBP/USD
* "run_example" = 18 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True, iterative_step_train, Lorenz_x+Lorenz_y
* "run_example" = 19 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = False, use_skip = True, use_condition = False, iterative_step_train
* "run_example" = 20 :
    experiment with outputs encoded in equal **bin_width** fashion, use_residual = True, use_skip = True, use_condition = True, iterative_step_train, GBP/USD
* "run_example" = 21 :
    experiment with outputs encoded in equal bin_count fashion, use_residual = True, use_skip = True, use_condition = True, iterative_step_train, GBP/USD (Regression)
* "run_example" = 22 :
    experiment with outputs encoded in equal bin_count fashion, 3 layer LSTM (Classification)
* "run_example" = 23 :
    experiment with outputs encoded in equal bin_count fashion, 3 layer LSTM (Regression)

To get started, open terminal and run:
`python3 main.py`