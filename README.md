# wavenet-finance

To configure these parameters, edit in args.json:
`nano args.json`

It is currently configured as:
```
{
    "test_round": 40,
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


To get started, open terminal and run:
`python3 main.py`