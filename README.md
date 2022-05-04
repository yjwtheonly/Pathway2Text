# Pathway2Text: Dataset and Method for Biomedical Pathway Description Generation

This is the implementation of our NAACL 2022 paper:

**Pathway2Text: Dataset and Method for Biomedical Pathway Description Generation**

Junwei Yang, Zequn Liu, Ming Zhang* and Sheng Wang*

https://openreview.net/pdf?id=SLQlZl3bHbc

Please cite our paper if you use this code.

### Download dataset: 

Our Pathway2Text dataset is released at https://zenodo.org/record/6510039#%23.Ym9F15NBz0o. Download ```mapping_database_to_pathway2text.json``` and ```pathway2text.json```,  put them in ```./finaldata/```.

### Download parameters: 

Our model with best performance is available at https://drive.google.com/file/d/1Whn9oZ0hIfOly0lIOBEPVafMn_CxSaly/view?usp=sharing. Download and put all the parameters in ```./params/```.

### Reproduce results:

 - For Graph2Text ：

```
  python graphtranswithdes.py --node-feat='labeldes' --used-part='graphdes'
```

 - For Text2Graph node classification:

```
  python nodeclassification.py --chosen-class='SIMPLE_CHEMICAL' --use-graph-des
```

Set ```--chosen-class='SIMPLE_CHEMICAL', 'MACROMOLECULE_MULTIMER' or 'MACROMOLECULE'``` for applying experiments on nodes of different type.

 - For Text2Graph link prediction:

```
  python linkprediction.py --use-graph-des --multiedge
```



The intermediate results are cached in the following paths:

```
  ./tokens/        --- tokenized sequences for node label, node des. and graph des
  ./embeddings/    --- [graph_des_embeddings, node_label_embeddings, node_des_embeddings] encoded by PLMs
  ./result/        --- generated descriptions for test graphs
```


