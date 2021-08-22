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
from evaluate_mAP import get_mAP
import re
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
def dropout_layer_factory():
    return Dropout(rate=0.2, name='dropout')



if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
if TRAIN_YOLO_TINY: TRAIN_MODEL_NAME += "_Tiny"


def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory() #calls add_layer_factory
            if insert_layer_name:
                new_layer._name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x) 
            
            
            #gotta replace new_layer with Add layer and some smart indexing manually
            
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)
    
    
def main():
    model = load_model('Tiny-yolo-object-resripdet-100ep.hdf5')
    print(model.summary())
    import sys
    sys.exit(1)

    global TRAIN_FROM_CHECKPOINT
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    trainset = Dataset('train')
    testset = Dataset('test')

    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    if TRAIN_TRANSFER:
        Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES)
    if TRAIN_FROM_CHECKPOINT:
        try:
            yolo.load_weights(TRAIN_FROM_CHECKPOINT)
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            TRAIN_FROM_CHECKPOINT = False

    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    #print(yolo.summary())
    
    import sys
    
    #l1 = yolo.get_layer('leaky_re_lu')
    # l1_layer = tf.keras.layers.Lambda(lambda x:x)(l1)
    # print(type(l1_layer))
    
    # for i in yolo.layers:
        # if i.name == 'batch_normalization':
            # l1_conv = tf.keras.layers.Conv2D(64,(3,3),activation = None,strides=(4,4), padding='same')(i)
    
    # l1_lambda = tf.keras.layers.Lambda(lambda x:x + 0)(l1.output)
    
    # #print(type(l1_lambda))
    
    # l1_conv = tf.keras.layers.Conv2D(64,(3,3),activation = None,strides=(4,4), padding='same')(l1_lambda)
    # l1_bn = tf.keras.layers.BatchNormalization()(l1_conv)
    # l2 = yolo.get_layer('batch_normalization_2').output
    # l2_add = tf.keras.layers.Add(name="l2_add")([l1_bn, l2])
    
    # m1 = Model(inputs = yolo.input, outputs=l2_add)
    # print(m1.output.shape)
    # print(type(m1.get_layer("l2_add")))
    #sys.exit(1)
    # print(type(l2_add))
    idx = 0
    for layer in yolo.layers:
        print(layer.name, idx)
        
        if 'batch_normalization_2' in layer.name:
            #new_layer = 
            #yolo.layers[idx + 1] = tf.keras.layers.Add(name="l2_add")([layer, yolo.layers[idx - 7]])
            conv = tf.keras.layers.Conv2D(64,(3,3),activation = None,strides=(4,4), padding='same')(yolo.layers[idx - 7].output)
            bn = tf.keras.layers.BatchNormalization()(conv)
            l = tf.keras.layers.Add(name="l2_add")([yolo.layers[idx].output, conv])
            l = yolo.layers[idx + 1](l)
            
            #print(yolo.layers[idx - 7].name)
            insert_layer_nonseq(yolo, '.*batch_normalization_2', add_layer_factory, 'add1_la_la')
        
        idx = idx + 1
        
    def add_layer_factory():
        return m1.get_layer("l2_add")
    
    #yolo_n = insert_layer_nonseq(yolo, '.*batch_normalization_2', add_layer_factory, 'add1_la_la')
    #yolo_n.save('temp_yolo_res.h5')
    
    
    
    
    
    # model = ResNet50()
    # model = insert_layer_nonseq(model, '.*conv1_relu.*', dropout_layer_factory, "bum_bum_tam_tam")

    # # Fix possible problems with new model
    # model.save('temp_resnet.h5')
    
    print(model.summary())
    sys.exit(1)
    
    import sys
    sys.exit(1)
    #optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.SGD()
    # yolo.save('temp-model.hdf5')
    # print(yolo.outputs)
    # t = tf.keras.models.load_model('temp-model.hdf5')
    # get_mAP(t, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    
    
    # import sys
    # sys.exit(1)
    
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

    mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES) # create second model to measure mAP

    best_val_loss = 1000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue
        
        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
            
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/count
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
        
        
        # if epoch % 5 == 0:
            # mAP_model.load_weights(save_directory) # use keras weights
            # get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
            
    # measure mAP of trained custom model
    mAP_model.load_weights(save_directory) # use keras weights
    get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    yolo.save('Tiny-yolo-object-data-aug-sgd-100ep.hdf5')
if __name__ == '__main__':
    main()
