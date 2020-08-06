ccg2lambda:
	cat ${input_path} | depccg_en --model elmo --format ccg2lambda --annotator spacy  > ${output_path}

sweep:
	wandb sweep trainer/sweep.yml
run:
	CUDA_VISIBLE_DEVICES=${gpu} wandb agent yummydum/lambda2text-src_trainer/${id}