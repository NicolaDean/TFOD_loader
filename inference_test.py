from load_model import SSD_MODEL

#TODO => SET DATASET FROM HERE SO WE HAVE AN EASY TO USE TRAINING/INFERENCE POINT
MODEL = SSD_MODEL(path="./mobilenet_ssd_config",model_name="ssd-mobilenet")

#DOWNLOAD all the configurations and dataset and checkpoits necessary for the model
MODEL.load_model()
#Get a working model ready to execute inference
most_recent_checkpoint_path = MODEL.get_most_recent_checkpoint_from_path('./training')
model,ssd_inference_fn      = MODEL.get_inference_model("./training/ckpt-37")

#Funziona solo dopo la prima inferenza dato che la backbone viene generata in fase di build del modello
model._feature_extractor.classification_backbone.summary()

#<class 'object_detection.models.ssd_mobilenet_v2_keras_feature_extractor.SSDMobileNetV2KerasFeatureExtractor'>
#<class 'object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch'>

#Start the working process on the model using the dataset and model selected
#MODEL.fine_tune_model()
