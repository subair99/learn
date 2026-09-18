FROM tensorflow/serving:2.14.0

COPY models/breast_cancer_model /models/breast_cancer_model/1
ENV MODEL_NAME="breast_cancer_model"