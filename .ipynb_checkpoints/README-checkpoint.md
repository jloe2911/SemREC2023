# SemREC2023

1. Download [OWL2Bench](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/ORE).
2. Download [ORE](https://github.com/semrec/semrec.github.io/tree/main/Datasets_SemREC2022/OWL2Bench).
3. Download [CaLiGraph](https://data.dws.informatik.uni-mannheim.de/CaLiGraph/CaLiGraph-for-SemREC/SemREC-2022-Datasets/).
4. Add the mentioned ontologies to a folder ```datasets```.
5. Run ```train.ipynb```. The notebook employs:

* Graph Attention Network (GAT).
* 2-Hop Graph Attention Network (2-Hop GAT): an extended version of Graph Attention Network that aggregates node information that are 2 hops away. 
* Filtered 2-Hop Graph Attention Network (2-Hop GAT): Filtered 2-Hop GAT with a filtering mechanism to include two rules.

6. Run ```eval.ipynb``` to evaluate the models. The metrics are Hits@1 and Hits@10. 