
#IMPORTANT => RUN conda activate XXXX if you dont want to ruin your current environment

#Clone the TFOD repository
if [ ! -d "./models" ]; then
    git clone --depth 1 https://github.com/tensorflow/models
    #Compile all protos config files
    protoc ./models/research/object_detection/protos/*.proto --python_out=.
    #Copy the generated setup in the working directory
    cp ./models/research/object_detection/packages/tf2/setup.py .
    #Install all the necessary python packages using setup.py
    python -m pip install .
fi

#PUT HERE YOUR DATASET FROM ROBOFLOW (or create one in local using tfrecord format)
#wget https://public.roboflow.com/ds/ZW4hFSBSW5?key=c0RwEMdb7Z -O roboflow.zip; unzip roboflow.zip; rm roboflow.zip

#Install some packages not included in the previous scripts
pip install -q roboflow
#pip install google-colab

#TEST IF INSTALLATION GONE WELL
python ./models/research/object_detection/builders/model_builder_tf2_test.py