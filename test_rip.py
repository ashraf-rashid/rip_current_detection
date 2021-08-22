#================================================================
#
#   File name   : train.py
#   Author      : PyLessons
#   Created date: 2020-08-06
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to train custom object detector
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import shutil
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import plot_model
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo, compute_loss
from yolov3.utils import load_yolo_weights
from yolov3.configs import *
from evaluate_mAP import get_mAP2
    
if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
if TRAIN_YOLO_TINY: TRAIN_MODEL_NAME += "_Tiny"

def main():
    global TRAIN_FROM_CHECKPOINT
    
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(f'GPUs {gpus}')
    # if len(gpus) > 0:
        # try: tf.config.experimental.set_memory_growth(gpus[0], True)
        # except RuntimeError: pass

    # if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    # writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    #trainset = Dataset('train')
    testset = Dataset('test')

    # steps_per_epoch = len(trainset)
    # global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    # warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    # total_steps = TRAIN_EPOCHS * steps_per_epoch

    # if TRAIN_TRANSFER:
        # Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        # load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    # yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES)
    # if TRAIN_FROM_CHECKPOINT:
        # try:
            # yolo.load_weights(TRAIN_FROM_CHECKPOINT)
        # except ValueError:
            # print("Shapes are incompatible, transfering Darknet weights")
            # TRAIN_FROM_CHECKPOINT = False

    # if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        # for i, l in enumerate(Darknet.layers):
            # layer_weights = l.get_weights()
            # if layer_weights != []:
                # try:
                    # yolo.layers[i].set_weights(layer_weights)
                # except:
                    # print("skipping", yolo.layers[i].name)
    
    # optimizer = tf.keras.optimizers.Adam()


    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            global_steps.assign_add(1)
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    #mAP_model = tf.keras.models.load_model('tiny_yolo_rip.hdf5') # load model to measure mAP
    #mAP_model =  tf.keras.models.load_model('tiny_yolo_rip_trained_for_100_epochs_with_standard_data_aug.hdf5')
    #print(mAP_model.outputs)
    
    # import sys
    # sys.exit(1)
    #mAP_model =  tf.keras.models.load_model('tiny_yolo_rip_trained_for_100_epochs_with_standard_data_aug.hdf5')
    #mAP_model = tf.keras.models.load_model('Tiny-yolo-object-twice-data-aug-resripdet1-100ep.hdf5')
    mAP_model = tf.keras.models.load_model('Tiny-yolo-object-resripdet-100ep.hdf5')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_resripdet1_twice_data_aug_100ep/yolov3_custom_Tiny_80_val_loss_   0.45')
    #mAP_model = tf.keras.models.load_model('Tiny-yolo-object-twice-data-aug-resripdet1-TL-coco.hdf5')
    #mAP_model = tf.keras.models.load_model('Tiny-yolo-object-tiny-yolo-denom-orig-by-4-100ep.hdf5')
    #mAP_model = tf.keras.models.load_model('temp-model-mish-activation.hdf5')
    #mAP_model.load_weights(r'D:\Imran_Razzak\TensorFlow-2.x-YOLOv3\checkpoints_resripdet1_mish_activation/yolov3_custom_Tiny_40_val_loss_   0.44')
    #mAP_model.load_weights(r'D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_resripdet1_final_activation_true/yolov3_custom_Tiny_80_val_loss_   0.44')
    #print(mAP_model.summary())
    
    #import sys
    #sys.exit(1)
    ##mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_rip_TL_coco/yolov3_custom_Tiny_64_val_loss_   0.44')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_best/yolov3_custom_Tiny_53_val_loss_   0.49')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_no_restart/yolov3_custom_Tiny_90_val_loss_   0.49')
   #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_more_warmup_highlr_0_001/yolov3_custom_Tiny_35_val_loss_   0.42')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_denom50_100ep/yolov3_custom_Tiny_15_val_loss_   0.52')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_denom_orig_by_4/yolov3_custom_Tiny_99_val_loss_   0.50')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_denom_orig_by_4/yolov3_custom_Tiny_20_val_loss_   0.55')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_obj_replacement/yolov3_custom_Tiny_65_val_loss_   0.48')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_obj_replacement/yolov3_custom_Tiny_65_val_loss_   0.48')
    #mAP_model.load_weights('D:/Imran_Razzak/TensorFlow-2.x-YOLOv3/checkpoints_tiny_yolo_denom_orig_by_4/yolov3_custom_Tiny_25_val_loss_   0.52')
    get_mAP2(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    #mAP_model.save('tiny_yolo_rip.hdf5')
    #mAP_model.save('tiny_yolo_rip_trained_for_30_epochs.hdf5')
    #mAP_model.save('tiny_yolo_rip_trained_for_1_epochs.hdf5') #=> this one is for 20 epochs
    
if __name__ == '__main__':
    main()