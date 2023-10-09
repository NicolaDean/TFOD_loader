import os
import sys

import pathlib
# directory reach
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + "/models/research")
print(directory + "/models/research")
#sys.path.append("/models/research")

from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import config_util
from models.research.object_detection.utils import visualization_utils as viz_utils
from models.research.object_detection.utils import colab_utils
from models.research.object_detection.builders import model_builder



from config_helper import *
from data_helper    import *

class SSD_MODEL():

    def __init__(self,path="./mobilenet_ssd_config", model_name="ssd-mobilenet",already_loaded_info=False):
        self.path       = path
        self.model_name = model_name
        pass

    def load_model(self):
        import shutil
        #CREARE O PULIRE LA CARTELLA in PATH
        if(not os.path.exists(self.path)):
            os.mkdir(self.path)
        else:
            shutil.rmtree(self.path)
            os.mkdir(self.path)

        rf,dataset,project = load_dataset("D5jpG7thd1uxwm3apfHd","jacob-solawetz","aerial-maritime")

        CONFIG = SSD_CONFIG(path=self.path)

        #Initialize model setting
        CONFIG.initialize_model_setting(self.model_name)
        #Download the .conf file containing all the relevant training/inference information to assemble the model
        CONFIG.download_config_file()
        #Download the pretrained model checkpoints weights
        CONFIG.download_base_model()
        #Set path to data
        CONFIG.set_dataset_paths(dataset,file_name ="movable-objects")
        #Generate a custom .conf file by adding informations like: dataset, train parameters....
        CONFIG.generate_custom_config_file()

        self.CONFIG = CONFIG

    def get_most_recent_checkpoint_from_path(self,path):
        import pathlib
        filenames = list(pathlib.Path(path).glob('*.index'))
        print(f'AAAAA:{os.listdir("./")}')
        filenames.sort()
        print(filenames)
        #generally you want to put the last ckpt from training in here
        model_dir = os.path.join(str(filenames[-1]).replace('.index',''))

        return model_dir
    
    def get_inference_model(self,checkpoint_path):
        
        #recover our saved model
        pipeline_config = self.CONFIG.pipeline_file
        
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint (TODO RESTORE)
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(checkpoint_path)


        def get_model_detection_function(model):
            """Get a tf.function for detection."""

            @tf.function
            def detect_fn(image):
                """Detect objects in image."""

                image, shapes   = model.preprocess(image)
                prediction_dict = model.predict(image, shapes)
                detections      = model.postprocess(prediction_dict, shapes)

                return detections, prediction_dict, tf.reshape(shapes, [-1])

            return detect_fn

        detect_fn = get_model_detection_function(detection_model)
        return detection_model, detect_fn, configs, model_config

