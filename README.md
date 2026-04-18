
# FSL-Live

FSL-Live is an open-source project focused on detecting and interpreting static hand gestures using machine learning. The system leverages the powerful TensorFlow framework to train and deploy a deep neural network model capable of recognizing various predefined actions.



## Getting Started

### Prerequisites

To get started with this project, follow these steps:

   On Windows: 

```sh
git clone https://github.com/ralphb420/FSL-Live.git

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

On Linux: 
```sh
git clone https://github.com/ralphb420/FSL-Live.git

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
To perform real-time hand gesture recognition, run either scripts:

- For using Scikit Trained Models:

    ```sh
    python realtime_predict.py
    ```

- For using Tensorflow Models:

    ```sh
    python realtime_predictH5_Inference.py
    ```
- [Senyas-FSL-Translator](https://github.com/antoineross/Senyas-FSL-Translator):

    ```sh
    python realtime_predictH5.py
    ```

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements

 - This projects' Tensorflow framework and training algorithm was inspired from [Senyas-FSL-Translator](https://github.com/antoineross/Senyas-FSL-Translator) made by [antoineross](https://github.com/antoineross).
 - [Data Set Used](https://www.kaggle.com/datasets/japorton/fsl-dataset)

