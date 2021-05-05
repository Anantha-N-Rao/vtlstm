import numpy as np
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, TimeDistributed,Flatten,Dropout,LSTM,Dense
from scipy.spatial.transform import Rotation as R
import random
from tensorflow.keras.callbacks import TensorBoard

tf.config.run_functions_eagerly(False)
num_classes = 6
input_shape = (256, 256, 2)
learning_rate = 0.0005
weight_decay = 0.00001
batch_size = 10
num_epochs = 1000
image_size = 256  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim =256
num_heads = 6
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4
mlp_head_units = [256, 128]

# batch=8
dir_path="E:/tartandataset/allfiles"
Trueposes=[]
Trueimages1=[]
Trueimages2=[]
z=tf.io.gfile.listdir(dir_path)     #environment
for i in range (len(z)):
    a=dir_path+'/'+z[i]
    easyorhard=tf.io.gfile.listdir(a)    #easy or hard
    for j in range (len(easyorhard)):
        ndir=a+'/'+easyorhard[0]
        pathtype=tf.io.gfile.listdir(ndir)       #paths
        for nj in range(len(pathtype)):
            images=[]
            poses=[]
            pathfiles=tf.io.gfile.listdir(ndir+'/'+pathtype[nj])
            openpose=open(ndir+'/'+pathtype[nj]+'/'+pathfiles[1],'r')
            poses=np.loadtxt(openpose)
            tp=poses[:,:3]
            rp=poses[:,3:] 
            euler_p=R.from_quat(rp)
            rotp=euler_p.as_euler('xyz', degrees=True)
            pose=np.append(tp,rotp,axis=1)
            pose=[t - s for s, t in zip(pose, pose[3:])]
            images=tf.io.gfile.listdir(ndir+'/'+pathtype[nj]+'/'+pathfiles[0])
            append_str = ndir+'/'+pathtype[nj]+'/'+pathfiles[0]+'/'
            images = [append_str + sub for sub in images]
            Trueposes=Trueposes+pose
            Trueimages1=Trueimages1+images[:-3]
            Trueimages2=Trueimages2+images[3:]
            
Trueposes=np.asarray(Trueposes)
Trueposes[:,[0,1]] = Trueposes[:,[1,0]]       
k=np.zeros((6,1))
meani=np.zeros((6,1))
stdi=np.zeros((6,1))
Tp=np.zeros((len(Trueposes),6))

for j in range(len(Trueposes)):
    if(Trueposes[j,5]<-300):
        Trueposes[j,5] = Trueposes[j,5]+360
    elif(Trueposes[j,5]>300):
        Trueposes[j,5] = Trueposes[j,5]-360

for i in range(6):
    k[i] = max(abs(Trueposes[:,i]))
    Tp[:,i] = Trueposes[:,i]/k[i]
    meani[i] = np.mean(Tp[:,i])
    stdi[i] = np.std(Tp[:,i])
    Tp[:,i] = (Tp[:,i]-meani[i])/stdi[i]



Tp = Tp.tolist()

def tf_diff_axis_0(a):
    return a[1:]-a[:-1]

def tf_diff_axis_1(a):
    return a[:,1:]-a[:,:-1]
            
def read_image_file(filename1):
    filename1 = tf.cast(filename1,tf.string)
    image_string1 = tf.io.read_file(filename1)
    image1 = tf.image.decode_jpeg(image_string1, channels=1)
    image1 = tf.image.convert_image_dtype(image1, tf.float32)
    image1 = tf.image.per_image_standardization(image1)
    image1 = tf.image.resize(image1, [256, 256])
    return tf.squeeze(image1)



def produce_data(xx1,xx2,zed): 
    image1=read_image_file(xx1)
    # if (np.random.random()<0.05):
    #     image2=image1
    #     zed=[0.0,0.0,0.0,0.0,0.0,0.0]
    # else:
    image2=read_image_file(xx2)
    # image2=read_image_file(xx2)
    images=tf.stack((image1,image2))
    images = tf.transpose(images,perm=[1,2,0])
    images=tf.squeeze(images)
    return images, zed

i1 = read_image_file(Trueimages1[909])
i2 = read_image_file(Trueimages2[909])
# i3 = read_image_file(Trueimages2[124])

n=120000
x1=Trueimages1[:n]
y1=Trueimages2[:n]
z1=Tp[:n]
x2=Trueimages1[80000:100000]
y2=Trueimages2[80000:100000]
z2=Tp[80000:100000]
x4=Trueimages1[:150]
y4=Trueimages2[:150]
z4=Tp[:150]
x5=[]
y5=[]
z5=[]
num_list = random.sample(range(0, 100000), 20000)
# for i in range(len(num_list)):
#     q = num_list[i]
#     x5.append(x1[q])
#     y5.append(y1[q])
#     z5.append(z1[q])


print("loaded data")
batch_size=20
def tf_dataset(x,y,z):
    dataset2 = tf.data.Dataset.from_tensor_slices((x,y,z))
    dataset2 = dataset2.shuffle(buffer_size = 100000,reshuffle_each_iteration=True)
    dataset3 = dataset2.take(2000)
    dataset3 = dataset3.map(produce_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset3
def tfs_dataset(x,y,z):
    dataset2 = tf.data.Dataset.from_tensor_slices((x,y,z))
    dataset2 = dataset2.shuffle(buffer_size = 1000)
    dataset2 = dataset2.take(batch_size*2)
    dataset2 = dataset2.map(produce_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset2
# zz = produce_data(x4[1],y4[1],z4[1])
def test_dataset(x,y,z):
    dataset2 = tf.data.Dataset.from_tensor_slices((x,y,z))
    # dataset2 = dataset2.shuffle(buffer_size = 100000,reshuffle_each_iteration=True)
    dataset3 = dataset2.take(batch_size*200)
    dataset3 = dataset3.map(produce_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset3
zz = tf_dataset(x1,y1,z1)
zz = zz.batch(batch_size)

valds = tfs_dataset(x2,y2,z2)
valds = valds.batch(batch_size)

newds = test_dataset(x4,y4,z4)
newds = newds.batch(batch_size)
lrelu = tf.keras.layers.LeakyReLU(0.001)
prelu = tf.keras.layers.PReLU()
iGl_uni = tf.keras.initializers.HeNormal()
ki=tf.keras.initializers.glorot_uniform()
Gl_uni = tf.keras.initializers.HeNormal()
fGl_uni = tf.keras.initializers.HeNormal()
lsGl_uni = tf.keras.initializers.Orthogonal()
l1 = tf.keras.regularizers.L1(0.0001)
l2 = tf.keras.regularizers.L2(0.0001)
l1l2 = tf.keras.regularizers.L1L2(0.01,0.01)
l1 = None
l2 = None
l1l2 =None

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation = 'relu', use_bias=True,kernel_initializer= Gl_uni)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
      
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

n, d = num_patches,projection_dim
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]


# Juggle the dimensions for the plot
# pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
# pos_encoding = tf.transpose(pos_encoding, (2,1,0))
# pos_encoding = tf.reshape(pos_encoding, (d, n))

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        # encoded = self.projection(patch)+pos_encoding
        return encoded
    



def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    input1,input2 = tf.split(inputs, 2, axis=-1)
    patches1 = Patches(patch_size)(input1)
    encoded_patches1 = PatchEncoder(num_patches, projection_dim)(patches1)
    patches2 = Patches(patch_size)(input2)
    encoded_patches2 = PatchEncoder(num_patches, projection_dim)(patches2)
    
    
    for i in range(transformer_layers):
    #leftside
        x1 = encoded_patches1
        attention_output1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, trainable=True,dropout=0, activity_regularizer=l1)(x1,x1,x1)
        x2 = attention_output1 + encoded_patches1
        # x2 = layers.Add()([attention_output1, encoded_patches1])
        x3 = layers.LayerNormalization()(x2)
        x4 = mlp(x3,hidden_units=transformer_units, dropout_rate=0)
        e1 = x3 + x4
        # e1 = layers.Add()([x3, x4])
        e1 = layers.LayerNormalization()(e1)
        encoded_patches1 = e1
        #rightside without attention
        
    for i in range(transformer_layers):
        x21 = encoded_patches2
        attention_output2 = layers.MultiHeadAttention(num_heads=num_heads,trainable=True, key_dim=projection_dim, dropout=0,activity_regularizer=l1)(encoded_patches2, encoded_patches2, encoded_patches2)
        x22 = attention_output2 + encoded_patches2
        x23 = layers.LayerNormalization()(x22)
        
        #rightside attention
        attention_output3 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0)(e1, e1, x23)
        x5 = attention_output3 + x23
        x6 = layers.LayerNormalization()(x5)
        
        x24 = mlp(x6, hidden_units=transformer_units, dropout_rate=0)
        e2 = x6 + x24
        e2 = layers.LayerNormalization()(e2)
        encoded_patches2 = e2
        # x7 = mlp(x6, hidden_units=transformer_units, dropout_rate=0.2)
        # encoded_patches2 = layers.Add()([x6, x7])
    # new_layer1 = Flatten()(encoded_patches1)
    # encoded_patches2 = layers.LayerNormalization()(encoded_patches2)
    new_layer3 = Flatten()(encoded_patches2)
    # new_layer3 = Dense(512, activation = None, use_bias=True, kernel_initializer= Gl_uni, kernel_regularizer=l1 )(new_layer2)
    # new_layer2 = layers.LayerNormalization()(new_layer2)
    
    # dense5 = Dense(512, use_bias = True, activation='sigmoid', kernel_initializer= Gl_uni)(new_layer3)
    # dense6 = Dense(512, use_bias = True, activation ='sigmoid',kernel_initializer= Gl_uni)(new_layer3)
    
    out1=Dense(6,activation=None,use_bias = True,kernel_initializer=fGl_uni,name="north")(new_layer3)
    # out2=Dense(1,activation=None, use_bias = True,kernel_initializer=fGl_uni,name="east")(new_layer3)
    # out3=Dense(1,activation=None,use_bias = True,kernel_initializer=fGl_uni,name="up")(new_layer3)
    # out4=Dense(1,activation=None, use_bias = True,kernel_initializer=fGl_uni,name="roll")(new_layer3)
    # out5=Dense(1,activation=None,use_bias = True,kernel_initializer=fGl_uni,name="pitch")(new_layer3)
    # out6=Dense(1,activation=None, use_bias = True,kernel_initializer=fGl_uni,name="yaw")(new_layer3)
    # ensemble_model = tf.keras.Model(inputs=inputs, outputs=[out1,out2,out3,out4,out5,out6])
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=[out1])

    return ensemble_model

trainloss=[]
trainmae=[]
testloss=[]
testmae=[]
physicsloss=[]
def run_experiment(model):
    with tf.device('/GPU:0'):
        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=learning_rate, weight_decay=weight_decay
        # )
        optimizer = tf.keras.optimizers.Adam(0.00005)
        @tf.function
        def north(targets,predictions):
            
            error_rot = tf.keras.losses.MSE(tf.transpose(targets),tf.transpose(predictions))
            # print('targets',targets)
            # print('predictions',predictions)
            # print('error',error_rot)
            return error_rot
    
        @tf.function
        def north_met(targets,predictions):
            # error_rot = tf.reduce_sum(tf.math.squared_differences(tf.transpose(targets),tf.transpose(predictions)),axis=0)
            # t1=tf.squeeze(targets[:,1])
            error_rot = tf.keras.losses.MAE(tf.transpose(targets),tf.transpose(predictions))
            # error_rot = tf.reduce_mean(error_rot)
            return error_rot
        
        @tf.function
        def custom_loss(targets,predictions):
            return north(targets,predictions)+0.5*north_met(targets,predictions)
        
        model.compile(
            optimizer=optimizer, loss = north, metrics = north_met
                )
    
        model.fit(zz, validation_data=valds, epochs=300, verbose=1)
        return model


vit_classifier = create_vit_classifier()
# vit_classifier.load_weights('largenetwork3')
# tryi = create_vit_classifier2()
# y=tf.transpose(vit_classifier.predict(zz.take(1)))
# print(y)
vit_classifier.summary()
history = run_experiment(vit_classifier)

tv = vit_classifier.trainable_variables
tshape=[]
for i in range(len(tv)):
    tshape.append(tf.shape(tv[i]))
    
ty=list(zz.take(1))
ty[0][1]
vit_classifier(ty[0][0])




imgs=ty[0][0]



imgs=tf.transpose(imgs, [0,3,1,2])
x1,x2 = tf_for_c_binary(imgs,10)

# tt=northloss(x1[7])
# tt2 = northloss(x2[7])
# plt.imshow(x1[9])
# plt.figure()
# plt.imshow(x2[9])

# tv=vit_classifier.trainable_variables
# z=tv[28]
# for i in range(3):
#     plt.figure()
#     plt.imshow(z[i])
    
# z=tv[36]
# for i in range(6):
#     plt.figure()
#     plt.imshow(z[i])

# for epoch in range(epochs):
#         print("\nStart of epoch %d" % (epoch,))
    
#         # Iterate over the batches of the dataset.
#         for step, (x_batch_train, y_batch_train) in enumerate(zz):
    
#             # Open a GradientTape to record the operations run
#             # during the forward pass, which enables auto-differentiation.
#             with tf.GradientTape() as tape:
    
#                 # Run the forward pass of the layer.
#                 # The operations that the layer applies
#                 # to its inputs are going to be recorded
#                 # on the GradientTape.
#                 logits = mlpmodel(x_batch_train, training=True) # Logits for this minibatch
#                 print(logits)
#                 print("loss")
#                 # print(logits)
#                 # print(tf.shape(x_batch_train))
    
#                 # Compute the loss value for this minibatch.
#                 # mlpmodel.addloss(lambda: 0.1*tf.reduce_mean((in1,in2,in3,in4)))
#                 loss_value = m_loss(y_batch_train, logits) 
#                 # loss_value2= 0.1*tf.reduce_mean(tf.keras.layers.Flatten()(x_batch_train))
#                 print(loss_value)
#                 # print(loss_value2)
    
#             # Use the gradient tape to automatically retrieve
#             # the gradients of the trainable variables with respect to the loss.
#             grads = tape.gradient(loss_value, mlpmodel.trainable_weights)
#             for layersshape in grads:
#                 # print(tf.shape(layersshape))
    
#             # Run one step of gradient descent by updating
#             # the value of the variables to minimize the loss.
#             optimizer.apply_gradients(zip(grads, mlpmodel.trainable_weights))
    
#             # Log every 200 batches.
#             if step % 200 == 0:
#                 print(
#                     "Training loss (for one batch) at step %d: %.4f"
#                     % (step, float(loss_value))
#                 )
#                 print("Seen so far: %s samples" % ((step + 1)))
