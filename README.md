# Topic-identification-for-spontaneous-Finnish-speech

Implementation of the approaches used in the paper: "Topic identification for spontaneous speech: Enriching audio features with embedded linguistic information".

The models are implemented using the SpeechBrain toolkit and the recipes are available in the subdirectories.

An overview of the explored Topic ID systems is given in the figure below:

<img src="topic_id_systems.png" width="600" height="500">

To run the experiments, you will need the following dependencies:
1. [SpeechBrain](https://speechbrain.github.io)
2. [Sklearn](https://scikit-learn.org/stable/)
3. [HyperPyYAML](https://pypi.org/project/HyperPyYAML/)
4. [JiWER](https://pypi.org/project/jiwer/)

To execute a recipe, you need to run the `train` and `hyperparams` files, for example:

`python ctc_aed_train.py ctc_aed_hyperparams.yaml`


Cite the paper:
`
@inproceedings{porjazovski2023topic,
  title={Topic Identification for Spontaneous Speech: Enriching Audio Features with Embedded Linguistic Information},
  author={Porjazovski, Dejan and Gr{\'o}sz, Tam{\'a}s and Kurimo, Mikko},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)},
  pages={396--400},
  year={2023},
  organization={IEEE}
}n
`
