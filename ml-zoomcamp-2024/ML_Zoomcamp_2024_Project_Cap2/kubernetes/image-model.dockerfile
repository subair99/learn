FROM tensorflow/serving:2.14.0

COPY models/brain_tumor_model /models/brain_tumor_model/1
ENV MODEL_NAME="brain_tumor_model"