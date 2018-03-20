## Cascaded Boundary Regression (CBR) for Temporal Action Detection
This is the repository for BMVC 2017 paper "Cascaded Boundary Regression for Temporal Action Detection". If you find this helps your research, please cite
      
    @inproceedings{gao2017cascaded,
      title={Cascaded Boundary Regression for Temporal Action Detection},
     author={Gao, Jiyang and Yang, Zhenheng and Nevatia, Ram},
     booktitle={BMVC},
     year={2017}
    }
    
The model is built using TensorFlow 0.12, dataset is THUMOS-14.

### Download unit-level features 
In this work, two-stream CNN features are used. The appearance features can be downloaded at here: [val set](), [test set](); the denseflow features can be downloaded here: [val set](https://drive.google.com/file/d/1-6dmY_Uy-H19HxvfK_wUFQCYHmlPzwFx/view?usp=sharing), [test set](https://drive.google.com/file/d/1Qm9lIJQFm5s6hDSB_2k1tj8q2tnabflJ/view?usp=sharing). Note that, val set is used for training, as the train set for THUMOS-14 does not contain untrimmed videos. 

### Action proposals
The test action proposals are provided in `test_proposals_from_TURN.txt`. If you want to generate your own proposals, please go to [TURN](https://github.com/jiyanggao/TURN-TAP) repository. 

### Train and test
Modify the feature paths in `main.py` and run `python main.py`. The test results are saved to `./eval/test_results/`. You need to run post-processing program to get the final detection results, `python postproc.py xxx.pkl`. The final detection results can be evaluated by THUMOS-14 official eval tool.  
