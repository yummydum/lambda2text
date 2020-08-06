ccg2lambda:
	# cat ${src} | depccg_en --format ccg2lambda --annotator spacy > ${trg}
	cat ${src} | depccg_en --format ccg2lambda --annotator spacy --model elmo  > ${trg}
