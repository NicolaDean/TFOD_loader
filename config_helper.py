import tarfile
import wget
from data_helper import get_data_path

##change chosen model to deploy different models available in the TF2 object detection zoo
MODELS_CONFIG = {
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d1': {
        'model_name': 'efficientdet_d1_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d2': {
        'model_name': 'efficientdet_d2_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
        'efficientdet-d3': {
        'model_name': 'efficientdet_d3_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
        'batch_size': 4
    },
        'ssd-mobilenet': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 16
    }
}

class SSD_CONFIG():

    def __init__(self,models_info=MODELS_CONFIG,num_steps=40000,path="./deploy"):
        self.MODELS_CONFIG = models_info
        self.path          = path
        self.num_steps     = num_steps

    def initialize_model_setting(self,chosen_model):
        self.model_name              = self.MODELS_CONFIG[chosen_model]['model_name']
        self.pretrained_checkpoint   = self.MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
        self.base_pipeline_file      = self.MODELS_CONFIG[chosen_model]['base_pipeline_file']
        self.batch_size              = self.MODELS_CONFIG[chosen_model]['batch_size'] #if you can fit a large batch in memory, it may speed up your training

        self.pipeline_fname         = f'{self.path}/{self.base_pipeline_file}' 
        self.fine_tune_checkpoint   = f'{self.path}/{self.model_name}/checkpoint/ckpt-0'

        return self.model_name, self.pretrained_checkpoint, self.base_pipeline_file, self.batch_size

    '''
    DOWNLOAD THE TFOD MODEL CONFIGURATION
    '''
    def download_config_file(self):
        download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + self.pretrained_checkpoint

        FILE_PATH   = f"{self.path}/{self.pretrained_checkpoint}"
        response = wget.download(download_tar, FILE_PATH)

        tar = tarfile.open(FILE_PATH)
        tar.extractall(self.path)
        tar.close()

    def set_dataset_paths(self,dataset,file_name="movable-objects"):
        test_record_fname, train_record_fname, label_map_pbtxt_fname = get_data_path(dataset=dataset,file_name=file_name)

        self.test_record_fname      = test_record_fname
        self.train_record_fname     = train_record_fname
        self.label_map_pbtxt_fname  = label_map_pbtxt_fname

        self.num_classes = self.get_num_classes(self.label_map_pbtxt_fname)


    '''
    DOWNLAOD THE TFOD PRETRAINED MODEL CHECKPOINTS
    '''
    def download_base_model(self):
        download_config = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/' + self.base_pipeline_file
        FILE_PATH   = f"{self.path}/{self.base_pipeline_file}"
        response = wget.download(download_config, FILE_PATH)

    def get_num_classes(self,pbtxt_fname):
        from models.research.object_detection.utils import label_map_util

        label_map       = label_map_util.load_labelmap(pbtxt_fname)
        categories      = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index  = label_map_util.create_category_index(categories)
        return len(category_index.keys())


    def generate_custom_config_file(self):
        #write custom configuration file by slotting our dataset, model checkpoint, and training parameters into the base pipeline file

        import re

        #%cd ./models/research/deploy
        print('writing custom configuration file')

        with open(self.pipeline_fname) as f:
            s = f.read()

        self.pipeline_file = f'{self.path}/pipeline_file.config'
        
        with open(self.pipeline_file, 'w') as f:

            # fine_tune_checkpoint
            s = re.sub('fine_tune_checkpoint: ".*?"',
                    'fine_tune_checkpoint: "{}"'.format(self.fine_tune_checkpoint), s)

            # tfrecord files train and test.
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(self.train_record_fname), s)
            s = re.sub(
                '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(self.test_record_fname), s)

            # label_map_path
            s = re.sub(
                'label_map_path: ".*?"', 'label_map_path: "{}"'.format(self.label_map_pbtxt_fname), s)

            # Set training batch_size.
            s = re.sub('batch_size: [0-9]+',
                    'batch_size: {}'.format(self.batch_size), s)

            # Set training steps, num_steps
            s = re.sub('num_steps: [0-9]+',
                    'num_steps: {}'.format(self.num_steps), s)

            # Set number of classes num_classes.
            s = re.sub('num_classes: [0-9]+',
                    'num_classes: {}'.format(self.num_classes), s)

            #fine-tune checkpoint type
            s = re.sub(
                'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

            f.write(s)

