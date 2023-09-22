# One Fits All: Power General Time Series Analysis by Pretrained LM (NeurIPS 2023 Spotlight)

Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin, "One Fits All: Power General Time Series Analysis by Pretrained LM,", NeurIPS, 2023. [[paper](https://arxiv.org/abs/2302.11939)]

The main challenge that blocks the development of pre-trained model for time series analysis is the lack of a large amount of data for training. In this work, we address this challenge by leveraging language or CV models, pre-trained from billions of tokens, for time series analysis. Specifically, we refrain from altering the self-attention and feedforward layers of the residual blocks in the pre-trained language or image model.

<div align="center"><img src=./pic/model_structure.png width=80% /></div>

## General Time Series Tasks

The proposed method outperforms other models on most tasks, including [long-term forecasting](./Long-term_Forecasting/README.md), [short-term forecasting](./Short-term_Forecasting/README.md), [classification](./Classification/README.md), [anomaly detection](./Anomaly_Detection/README.md), [imputation](./Imputation/README.md), and [few-shot leanring](./Few-shot_Learning/README.md), [zero-short learning](./Zero-shot_Learning/README.md).

<div align="center"><img src=./pic/main_result.png width=60% /></div>

## Get Start

- Install Python>=3.8, PyTorch 1.8.1.
- Follow the instructions provided in the respective task folder.


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{zhou2023onefitsall,
  title={{One Fits All}: Power General Time Series Analysis by Pretrained LM},
  author={Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin},
  booktitle={NeurIPS},
  year={2023}
}
```

## Further Reading
Survey on Transformers in Time Series:

Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. "Transformers in time series: A survey.", IJCAI, 2023. [[paper](https://arxiv.org/abs/2202.07125)]


## Contact

If you have any question or want to use the code, please contact tian.zt@alibaba-inc.com or niupeisong.nps@alibaba-inc.com .

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/DAMO-DI-ML/ICML2022-FEDformer

https://github.com/thuml/Time-Series-Library

https://github.com/gzerveas/mvts_transformer