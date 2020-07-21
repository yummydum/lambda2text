setup:
	python -m spacy download en
	depccg_en download elmo

ccg2lambda:
	# cat data/glue_split/${file} | depccg_en --format ccg2lambda --annotator spacy > data/formal_split/${file}
	cat data/glue_split/${file} | depccg_en --model elmo --format ccg2lambda --annotator spacy --gpu ${gpu} > data/formal_split/${file}


fit:	
	dir = data/translation
	onmt_preprocess -train_src dir/src-train.txt -train_tgt dir/tgt-train.txt -valid_src dir/src-val.txt -valid_tgt dir/tgt-val.txt -save_data data/result
	onmt_train -data data/translation/ -save_model checkpoint/model

tranlate:
	onmt_translate -model demo-model_step_100000.pt -src data/translation/src-val.txt -output pred.txt -replace_unk -verbose
