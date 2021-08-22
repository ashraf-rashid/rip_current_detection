# YOLO options
YOLO_TYPE                   = "yolov3" # yolov4 or yolov3
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_V3_WEIGHTS             = "model_data/yolov3.weights"
YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolov3-tiny.weights"
YOLO_V4_TINY_WEIGHTS        = "model_data/yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = False # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
# Train options
TRAIN_YOLO_TINY             = True
TRAIN_SAVE_BEST_ONLY        = False # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = True # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES               = "model_data/rip/rip.names"#"model_data/rip/rip.names"#"mnist/mnist.names"#"model_data/rip/rip.names"
TRAIN_ANNOT_PATH            = "model_data/rip/rip_train_with_augmentation_new.txt"#"model_data/coco/train2017.txt"#"model_data/rip/rip_train_with_augmentation_new.txt" #"mnist/mnist_train.txt"#"model_data/rip/rip_train_object_crop.txt" 
TRAIN_LOGDIR                = "./log"
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints_resripdet1_mish_activation"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}_custom"
TRAIN_LOAD_IMAGES_TO_RAM    = False # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 8 #works till batch size 40
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False#"checkpoints_tiny_yolo_best/yolov3_custom_Tiny_52_val_loss_   0.48" #False # "checkpoints/yolov3_custom"
TRAIN_LR_INIT               = 1e-4#1e-4 #1e-2
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2#8#4#2
TRAIN_EPOCHS                = 100#100#150 #30#100 #50 for twice data aug

# TEST options
TEST_ANNOT_PATH             = "model_data/rip/rip_test_new.txt"#"model_data/rip/rip_validation_set.txt"#"model_data/coco/val2017.txt"#"model_data/rip/rip_test_new.txt"#"model_data/rip/rip_validation_set.txt"#"model_data/rip/rip_test_new.txt"#"model_data/rip/rip_validation_set.txt"#"mnist/mnist_test.txt"#"model_data/rip/rip_validation_set.txt"
TEST_BATCH_SIZE             = 8 #4
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45


#YOLOv3-TINY and YOLOv4-TINY WORKAROUND
if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32, 64]    
    YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
                               [[81,  82], [135, 169], [344, 319]],
                               [[0,    0], [0,     0], [0,     0]]]