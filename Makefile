ccg2lambda:
	cat ${src} | depccg_en --format ccg2lambda --annotator spacy --model elmo  > ${trg}

ccg2lambda_xml:
	cat ${src} | depccg_en --format jigg_xml_ccg2lambda --annotator spacy --model elmo  > ${trg}
	# cat sentences.txt | depccg_en --format jigg_xml_ccg2lambda --annotator spacy --model elmo  > sentences.sem.xml                  
	# python scripts/prove.py sentences.sem.xml --graph_out graphdebug.html