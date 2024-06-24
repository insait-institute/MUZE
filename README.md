## Structure

The project contains three main parts: 
* Data Gathering and Analysis (get_and_analyse_data folder)
* Model Training and Evaluation (model_and_finetuning folder)
* Multiple Model Bases as in OpenCLIP (model_configure folder)

Inside model_and_finetuning, we find 
* <scripts> used to train and evaluate
* <src> MUZE and CLIP finetuned structure and python code
* <finetuning_and_inference> Experiments finetuning and inference

## Dataset

* The dataset is containing about 200.000 museum exhibits, in the form of image/text pairs. In the files, you can find the links for the images as well as multiple columns denoting the different attributes associated with the exhibits. The attributes' values of each exhibit is stored as a number and mapped to the corresponding value in each attribute/class (see classes.json)
* The MUZE dataset archive is available at: https://drive.google.com/file/d/1cZfkfV8inrSSVgdfs5f2ChoFrSrwaEZZ/view?usp=sharing
* The archive is structured pe each museum, having train, test, and val splits for the data. The ratio for train, test and val are about: 70:20:10
* You can see images, attributes and values associated with the exhibits in the figure below:
<img src="https://github.com/AstridMocanu/MUZE/blob/main/figures/dataset_presentation-transparent.png?raw=true" width="800">



## Method Architecture

* We worked with the MUZE dataset using the MUZE method, which was based on CLIP-like encoders and Transformer tops. The following is a schematic representation of our proposed method (MUZE).
* We show the process of obtaining CLIP embeddings for the input image (eI), attribute names (eAi) and attribute values (eVi).
* After replacing the embeddings of the query attribute values with [MASK] tokens we pass the obtained sequence of embeddings through parseNet to obtain the predicted embeddings for the query attributes.
* The CLIP Image Encoder and parseNet are trained to maximize the cosine similarity between the target and predicted embeddings.
<img src="https://github.com/AstridMocanu/MUZE/blob/main/figures/method_diagram-transparent4.png?raw=true" width="800">

## Paper
* For more details about the project and experiments see: 
