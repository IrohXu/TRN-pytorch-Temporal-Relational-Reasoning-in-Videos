# TRN-pytorch-Temporal-Relational-Reasoning-in-Videos
### Introduction  
Implementation for Temporal Relational Reasoning in Videos. This is a NYU course project for DS-GA 3001.004/.005 Introduction to Computer Vision (Spring 2021)  
In this project, I will implement two models: (1) Long-term Recurrent Convolutional Networks for Visual Recognition and Description; (2) Temporal Relational Reasoning in Videos. The report will concentrate in discussing MIT CSAIL's creative work: Temporal Relational Reasoning in Videos.  


### Data preparation  

For this implementation, I only test one dataset from Temporal Relational Reasoning in Videos: Jester. The link of the dataset is here:  [Jester](https://20bn.com/datasets/jester) If you can not download the dataset due to web problems, I suggest you download it from kaggle. Here is the link: [kaggle-jester1](https://www.kaggle.com/zhaochengdu1998/jester1) [kaggle-jester2](https://www.kaggle.com/zhaochengdu1998/jester2)  
  
To simplify the experiment process, I use a subset of Jester as a mini version: mini-Jester, you can download mini-Jester here: [mini-Jester](https://pan.baidu.com/s/1_2RiBQKiuPwumV6ujI6BHg), download code: x0c6

### Training and Testing

### Reference:
B. Zhou, A. Andonian, and A. Torralba. Temporal Relational Reasoning in Videos. European Conference on Computer Vision (ECCV), 2018. [PDF](https://arxiv.org/pdf/1711.08496.pdf)
```
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```  
Donahue, J., Anne Hendricks, L., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., & Darrell, T. Long-term recurrent convolutional networks for visual recognition and description. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2015 [PDF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)
```
@article{donahue2015long,
  title={Long-term recurrent convolutional networks for visual recognition and description},
  author={Donahue, Jeffrey and Anne Hendricks, Lisa and Guadarrama, Sergio and Rohrbach, Marcus and Venugopalan, Subhashini and Saenko, Kate and Darrell, Trevor},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2625--2634},
  year={2015}
}
``` 
Part of my code is refer to below:  
https://github.com/MRzzm/action-recognition-models-pytorch  
https://github.com/zhoubolei/TRN-pytorch  


