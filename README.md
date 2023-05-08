# CoMSA: A Modeling Driven Sampling Approach
CoMSA is a modeling driven sampling approach for configuration performance testing.
CoMSA is designed in scenarios where there does not exist historical performance testing results (code start) and when there exists historical information (warm start).
# Implementation
This repo is an offline version, which have run all the configuration performance in advance, and simulate the online procedure.
If you would like to test a online version, you can use the "history_running.py" as a pattern.

The "code+allPerf" folder contain the code files of the approach and two ablation experiments. Respectively, the implementation files are "main.py", "main_ablation1.py" and "main_ablation2.py".

We test our approach in four subject, to switch the subjects, please modify the subject name and number in main function, you could directly follow the comment in code:

The detail of modifications:

[SubjectName]=[lrzip, llvm, x264, sqlite]

1. Functions:

    generate_config_[SubjectName]

    transfer_config_[SubjectName]

2. File:

    perf_first_[SubjectName].csv
3. variables:

    stop_point: configuration number that needed to test in each new commit
    
    init_length: the length of initial set in selector initialization in cold-start scenario
    
    commit_number: the total number of the commit

The "Compare" folder contain the raw files and data files of regression methods and sampling approach that we have tested.

The "Perf-analysis" folder contain the scripts of our evaluations and results.