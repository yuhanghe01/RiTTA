CUDA_VISIBLE_DEVICES=0 python inference_extra.py --original_args="finetuned_model_dir/summary.jsonl" \
--model="finetuned_model_dir/pytorch_model_2.bin" --num_steps 200 --guidance 3 --num_samples 1
