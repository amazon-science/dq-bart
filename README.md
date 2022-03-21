## DQ-BART
This repository contains the authors' implementation of the ACL 2022 paper "[DQ-BART: Efficient Sequence-to-Sequence Model via
Joint Distillation and Quantization](https://assets.amazon.science/a9/e4/4430bc9d4b35aa33076a9ec3aa02/dq-bart-efficient-sequence-to-sequence-model-via-joint-distillation-and-quantization.pdf)."

## Requirements
- Install PyTorch from the [official website](https://pytorch.org/get-started/locally/).
- Install dependencies via `pip install -r requirements.txt`. 
- The teacher model should be available locally, e.g., downloading manually from the [huggingface model hub](https://huggingface.co/models).

## Sample Command
- The following command will train an `8-8-8 3-1` model on CNN/DailyMail dataset. You should adjust `model_name_or_path` and `dataset_name` accordingly for other tasks. You may use [accelerate](https://github.com/huggingface/accelerate) for distributed training. 
    ```bash
    python3 run_summarization_no_trainer.py 
      --model_name_or_path bart-base-cnn 
      --dataset_name cnn_dailymail 
      --dataset_config_name 3.0.0 
      --pred_distill 
      --intermediate_distill 
      --num_train_epochs 20 
      --weight_bits 8 
      --do_train 
      --do_test 
      --distill_encoder 3 
      --distill_decoder 1 
      --learning_rate 3e-5
    ```
## Citation
You may cite our work using
```
@inproceedings{li2022dqbart,
  author={Li, Zheng and Wang, Zijian and Tan, Ming and Nallapati, Ramesh and Bhatia, Parminder and Arnold, Andrew and Xiang, Bing and Roth, Dan},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  title={DQ-BART: Efficient Sequence-to-Sequence Model via Joint Distillation and Quantization},
  url={https://assets.amazon.science/a9/e4/4430bc9d4b35aa33076a9ec3aa02/dq-bart-efficient-sequence-to-sequence-model-via-joint-distillation-and-quantization.pdf},
  year={2022}
}
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
