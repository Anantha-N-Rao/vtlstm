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

tf.config.run_functions_eagerly(True)
num_classes = 6
input_shape = (256, 256, 2)
learning_rate = 0.0005
weight_decay = 0.00001
batch_size = 10
num_epochs = 1000
image_size = 256  # We'll resize input images to this size
patch_size = 32  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim =128
num_heads = 6
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 6
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
        ndir=a+'/'+easyorhard[j]
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
Tp=np.zeros((len(Trueposes),6))
# Set target range between -1 and 1
# for i in range(0,6,2):
#     k[i]=max(abs(Trueposes[:,i]))
#     k[2*i] = max(abs(Trueposes[:,i]))
#     for j in range(len(Trueposes)):
#         if(Trueposes[j,0]<0):
#             Tp[j,0] = abs(Trueposes[j,0])
#         elif(Trueposes[j,0]>0):
#             Tp[j,1] = abs(Trueposes[j,0])
#         if(Trueposes[j,1]<0):
#             Tp[j,2] = abs(Trueposes[j,1])
#         elif(Trueposes[j,1]>0):
#             Tp[j,3] = abs(Trueposes[j,1])
#         if(Trueposes[j,2]<0):
#             Tp[j,4] = abs(Trueposes[j,2])
#         elif(Trueposes[j,2]>0):
#             Tp[j,5] = abs(Trueposes[j,2])
#         if(Trueposes[j,3]<0):
#             Tp[j,6] = abs(Trueposes[j,3])
#         elif(Trueposes[j,3]>0):
#             Tp[j,7] = abs(Trueposes[j,3])
#         if(Trueposes[j,4]<0):
#             Tp[j,8] = abs(Trueposes[j,4])
#         elif(Trueposes[j,4]>0):
#             Tp[j,9] = abs(Trueposes[j,4])
for j in range(len(Trueposes)):
    if(Trueposes[j,5]<-300):
        Trueposes[j,5] = Trueposes[j,5]+360
    elif(Trueposes[j,5]>300):
        Trueposes[j,5] = Trueposes[j,5]-360
#         if(Trueposes[j,5]<0):
#             Tp[j,10] = abs(Trueposes[j,5])
#         elif(Trueposes[j,5]>0):
#             Tp[j,11] = Trueposes[j,5]

for i in range(6):
    k[i] = max(abs(Trueposes[:,i]))
    # meani = np.mean(Tp[:,i])
    # stdi = np.std(Tp[:,i])
    Tp[:,i] = Trueposes[:,i]/k[i]
    meani = np.mean(Tp[:,i])
    stdi = np.std(Tp[:,i])
    Tp[:,i] = (Tp[:,i]-meani)/stdi
    

Tp = Tp.tolist()

def tf_diff_axis_0(a):
    return a[1:]-a[:-1]

def tf_diff_axis_1(a):
    return a[:,1:]-a[:,:-1]
            
def read_image_file(filename1):
    image_string1 = tf.io.read_file(filename1)
    image1 = tf.image.decode_jpeg(image_string1, channels=1)
    image1 = tf.image.convert_image_dtype(image1, tf.float32)
    image1 = tf.image.per_image_standardization(image1)
    image1 = tf.image.resize(image1, [256, 256])
    return tf.squeeze(image1)



def produce_data(xx1,xx2,zed): 
    image1=read_image_file(xx1)
    image2=read_image_file(xx2)
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
for i in range(len(num_list)):
    q = num_list[i]
    x5.append(x1[q])
    y5.append(y1[q])
    z5.append(z1[q])


print("loaded data")
# batch_size=20
def tf_dataset(x,y,z):
    dataset2 = tf.data.Dataset.from_tensor_slices((x,y,z))
    dataset2 = dataset2.shuffle(buffer_size = 100000,reshuffle_each_iteration=True)
    dataset3 = dataset2.take(20000)
    dataset3 = dataset2.map(produce_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset3
def tfs_dataset(x,y,z):
    dataset2 = tf.data.Dataset.from_tensor_slices((x,y,z))
    dataset2 = dataset2.shuffle(buffer_size = 1000)
    dataset2 = dataset2.take(batch_size*5)
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
l1 = tf.keras.regularizers.L1(0.01)
l2 = tf.keras.regularizers.L2(0.001)
l1l2 = tf.keras.regularizers.L1L2(0.01,0.01)
# l1 = None
# l2 = None
l1l2 =None

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation = 'sigmoid', use_bias=True,kernel_initializer= Gl_uni, kernel_regularizer = l1)(x)
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
        attention_output1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, trainable=True,dropout=0.25, activity_regularizer=l1)(x1,x1,x1)
        x2 = layers.Add()([attention_output1, encoded_patches1])
        x3 = layers.LayerNormalization()(x2)
        x4 = mlp(x3,hidden_units=transformer_units, dropout_rate=0)
        e1 = layers.Add()([x3, x4])
        e1 = layers.LayerNormalization()(e1)
        encoded_patches1 = e1
        #rightside without attention
        x21 = encoded_patches2
        attention_output2 = layers.MultiHeadAttention(num_heads=num_heads,trainable=True, key_dim=projection_dim, dropout=0.25,activity_regularizer=l1)(e1, e1, x21)
        x22 = layers.Add()([attention_output2, encoded_patches2])
        x23 = layers.LayerNormalization()(x22)
        x24 = mlp(x23, hidden_units=transformer_units, dropout_rate=0)
        e2 = layers.Add()([x23, x24])
        e2 = layers.LayerNormalization()(e2)
        encoded_patches2 = e2
        #rightside attention
        # attention_output3 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.2)(e1, e1, x23)
        # x5 = layers.Add()([attention_output3, x23])
        # x6 = layers.LayerNormalization()(x5)
        
        # x7 = mlp(x6, hidden_units=transformer_units, dropout_rate=0.2)
        # encoded_patches2 = layers.Add()([x6, x7])
    # new_layer1 = Flatten()(encoded_patches1)
    # encoded_patches2 = layers.LayerNormalization()(encoded_patches2)
    new_layer2 = Flatten()(encoded_patches2)
    new_layer3 = Dense(512, activation = 'sigmoid', use_bias=True, kernel_initializer= Gl_uni, kernel_regularizer=l1 )(new_layer2)
    # new_layer2 = layers.LayerNormalization()(new_layer2)
    
    dense5 = Dense(512, use_bias = True, activation='sigmoid', kernel_initializer= Gl_uni,kernel_regularizer=l2)(new_layer3)
    dense6 = Dense(512, use_bias = True, activation ='sigmoid',kernel_initializer= Gl_uni, kernel_regularizer=l2)(new_layer3)
    # dense5 = Dense(512, use_bias = True, activation=lrelu, kernel_regularizer=l2)(dense5)
    # dense6 = Dense(512, use_bias = True, activation =lrelu, kernel_regularizer=l2)(dense6)
    # Classify outputs.
    # out1=Dense(1,activation='sigmoid',use_bias = True,kernel_initializer=fGl_uni,name="northneg")(dense5)
    # out2=Dense(1,activation='sigmoid',use_bias = True,kernel_initializer=fGl_uni,name="north")(dense5)
    # out3=Dense(1,activation='sigmoid', use_bias = True,kernel_initializer=fGl_uni,name="eastneg")(dense5)
    # out4=Dense(1,activation='sigmoid', use_bias = True,kernel_initializer=fGl_uni,name="east")(dense5)
    # out5=Dense(1,activation='sigmoid',use_bias = True,kernel_initializer=fGl_uni,name="upneg")(dense5)
    # out6=Dense(1,activation='sigmoid',use_bias = True,kernel_initializer=fGl_uni,name="up")(dense5)
    # out7=Dense(1,activation='sigmoid', use_bias = True,kernel_initializer=fGl_uni,name="rollneg")(dense6)
    # out8=Dense(1,activation='sigmoid', use_bias = True,kernel_initializer=fGl_uni,name="roll")(dense6)
    # out9=Dense(1,activation='sigmoid',use_bias = True,kernel_initializer=fGl_uni,name="pitchneg")(dense6)
    # out10=Dense(1,activation='sigmoid',use_bias = True,kernel_initializer=fGl_uni,name="pitch")(dense6)
    # out11=Dense(1,activation='sigmoid', use_bias = True,kernel_initializer=fGl_uni,name="yawneg")(dense6)
    # out12=Dense(1,activation='sigmoid', use_bias = True,kernel_initializer=fGl_uni,name="yaw")(dense6)
    
    # out1=Dense(1,activation=None,use_bias = True,kernel_initializer=fGl_uni,name="northneg")(dense5)
    # out1=Dense(1,activation='tanh',use_bias = True,kernel_initializer=fGl_uni,name="north")(dense5)
    # out2=Dense(1,activation='tanh', use_bias = True,kernel_initializer=fGl_uni,name="east")(dense5)
    # out3=Dense(1,activation='tanh',use_bias = True,kernel_initializer=fGl_uni,name="up")(dense5)
    # out4=Dense(1,activation='tanh', use_bias = True,kernel_initializer=fGl_uni,name="roll")(dense6)
    # out5=Dense(1,activation='tanh',use_bias = True,kernel_initializer=fGl_uni,name="pitch")(dense6)
    # out6=Dense(1,activation='tanh', use_bias = True,kernel_initializer=fGl_uni,name="yaw")(dense6)
    
    out1=Dense(1,activation=None,use_bias = True,kernel_initializer=fGl_uni,name="north")(dense5)
    out2=Dense(1,activation=None, use_bias = True,kernel_initializer=fGl_uni,name="east")(dense5)
    out3=Dense(1,activation=None,use_bias = True,kernel_initializer=fGl_uni,name="up")(dense5)
    out4=Dense(1,activation=None, use_bias = True,kernel_initializer=fGl_uni,name="roll")(dense6)
    out5=Dense(1,activation=None,use_bias = True,kernel_initializer=fGl_uni,name="pitch")(dense6)
    out6=Dense(1,activation=None, use_bias = True,kernel_initializer=fGl_uni,name="yaw")(dense6)
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=[out1,out2,out3,out4,out5,out6])
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
            return error_rot
    
        @tf.function
        def north_met(targets,predictions):
            # error_rot = tf.reduce_sum(tf.math.squared_differences(tf.transpose(targets),tf.transpose(predictions)),axis=0)
            # t1=tf.squeeze(targets[:,1])
            error_rot = tf.keras.losses.MAE(tf.transpose(targets),tf.transpose(predictions))
            # error_rot = tf.reduce_mean(error_rot)
            return error_rot

        model.compile(
            optimizer=optimizer
                )
    

        final_loss=[]
        # epochs=5
        @tf.function
        def geom_c(ref,im):
            a=np.sum(np.multiply(ref,im))
            b=np.sum(im)
            return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        
        def geom_cen(ref, im):
            return tf.numpy_function(geom_c, [ref,im], [tf.float32])
        
        def contrast_binary(image):
            thresh = 0.2
            # image = list(image.numpy())
            i1=np.zeros((256,256), dtype=np.float32)
            i2=np.zeros((256,256), dtype=np.float32)
            # print(tf.shape(image))
            ih=np.diff(image,axis=1)
            iv=np.diff(image,axis=0)
            i1[:,:-1] = ih
            i2[:-1,:] = iv
            i1[abs(i1)>=thresh]=1
            i2[abs(i2)>=thresh]=1
            i1[abs(i1)<thresh]=0
            i2[abs(i2)<thresh]=0
            return tf.cast((np.float32(i1+i2)), dtype = tf.float32)
            # return tf.cast((i1+i2), dtype = tf.float32)
        def tf_c_binary(image):
            return tf.numpy_function(contrast_binary, [image], [tf.float32])
        
        def sumtensor(tensor):
            return np.mean(tensor, axis=0, dtype=np.float32)
        
        def tf_sumtensor(tensor):
            return tf.numpy_function(sumtensor, [tensor], [tf.float32])
        
        def lossim(i1,i2):
            ixs=np.subtract(i1,i2)
            return ixs/np.max(np.abs(ixs))
            # return np.sum(np.sign(ixs))
        
        def tf_lossim(i1,i2):
            return tf.numpy_function(lossim, [i1,i2], [tf.float32])
        
        def trfunc():
            tiled=tf.range(1/128,1+1/128,1/128, dtype=tf.float32)
            tilen=tf.range(1,0,-1/128, dtype=tf.float32)
            txpos=tf.tile(tiled,[128])
            txneg=tf.tile(tilen,[128])
            txpos=tf.split(txpos, 128, axis=0)
            txneg=tf.split(txneg, 128, axis=0)
            typos=tf.transpose(tf.split((tf.tile(tiled,[128])),128,axis=0))
            tyneg=tf.transpose(tf.split((tf.tile(tilen,[128])),128,axis=0))
            tr4=tf.norm((txpos,typos),axis=0)
            tr1=tf.norm((txneg,tyneg), axis=0)
            tr3=tf.norm((txneg,typos), axis=0)
            tr2=tf.norm((txpos,tyneg), axis=0)
            tr1=tf.math.add(tr1,txneg)
            tr2=tf.math.add(tr2,txpos)
            tr3=tf.math.add(tr3,txneg)
            tr4=tf.math.add(tr4,txpos)
            tr1=tr1/tf.reduce_max(tr1)
            tr2=tr2/tf.reduce_max(tr2)
            tr3=tr3/tf.reduce_max(tr3)
            tr4=tr4/tf.reduce_max(tr4)
            return tr1,tr2,tr3,tr4
        
        def rollloss(img1):
            tiled=tf.range(1/128,1+1/128,1/128, dtype=tf.float32)
            tilen=tf.range(1,0,-1/128, dtype=tf.float32)
            txpos=tf.tile(tiled,[128])
            txneg=tf.tile(tilen,[128])
            txpos=tf.split(txpos, 128, axis=0)
            txneg=tf.split(txneg, 128, axis=0)
            typos=tf.transpose(tf.split((tf.tile(tiled,[128])),128,axis=0))
            tyneg=tf.transpose(tf.split((tf.tile(tilen,[128])),128,axis=0))
            tr4=tf.norm((txpos,typos),axis=0)
            tr1=tf.norm((txneg,tyneg), axis=0)
            tr3=tf.norm((txneg,typos), axis=0)
            tr2=tf.norm((txpos,tyneg), axis=0)
            # img1=tf_c_binary(img1)
            # print(img1)
            i1 = img1[:128,:128]
            ir1=geom_cen(tr2,img1[:128,:128])
            ir2=geom_cen(tr4,img1[:128,128:256])
            ir3=geom_cen(tr1,img1[128:256,:128])
            ir4=geom_cen(tr3,img1[128:256,128:256])
            return ir1,ir2,ir3,ir4
    
        def northloss(img1):
            tiled=tf.range(1/128,1+1/128,1/128, dtype=tf.float32)
            tilen=tf.range(1,0,-1/128, dtype=tf.float32)
            txpos=tf.tile(tiled,[128])
            txneg=tf.tile(tilen,[128])
            txpos=tf.split(txpos, 128, axis=0)
            txneg=tf.split(txneg, 128, axis=0)
            typos=tf.transpose(tf.split((tf.tile(tiled,[128])),128,axis=0))
            tyneg=tf.transpose(tf.split((tf.tile(tilen,[128])),128,axis=0))
            tr4=(txpos+typos)/2
            tr1=(txneg+tyneg)/2
            tr2=(txpos+tyneg)/2
            tr3=(typos+txneg)/2
            # tr4=tf.norm((txpos,typos),axis=0)
            # tr1=tf.norm((txneg,tyneg), axis=0)
            # tr3=tf.norm((txneg,typos), axis=0)
            # tr2=tf.norm((txpos,tyneg), axis=0)
            
            
            ix1,ix2 = tf.split(img1, 2, 1)
            # iyy = tf.split(img1, 2, 0)
            i1,i3 = tf.split(ix1, 2, 0)
            i2,i4 = tf.split(ix2, 2, 0)
            ir1=geom_cen(tr1,i1)
            ir2=geom_cen(tr2,i2)
            ir3=geom_cen(tr3,i3)
            ir4=geom_cen(tr4,i4)
            return -ir1,-ir2,-ir3,-ir4
        
    
        def pitchloss(img1):
            tr1,tr2,tr3,tr4=trfunc()
            # img1=tf_c_binary(img1)
            ir1=geom_cen(tr1,img1[:128,:128])
            ir2=geom_cen(tr2,img1[:128,128:256])
            ir3=geom_cen(tr4,img1[128:256,:128])
            ir4=geom_cen(tr3,img1[128:256,128:256])
            return -ir1,-ir2,ir3,ir4
        
        
        
            
        def yawloss(img1):
            tr1,tr2,tr3,tr4=trfunc()
            # img1=tf_c_binary(img1)
            ir1=geom_cen(tr1,img1[:128,:128])
            ir2=geom_cen(tr2,img1[:128,128:256])
            ir3=geom_cen(tr3,img1[128:256,:128])
            ir4=geom_cen(tr4,img1[128:256,128:256])
            return -ir1,ir2,-ir3,ir4
        
        def simple_relu(x):
            if tf.greater(x, 0.0):
                return x
            else:
                return 0.0
        
        # `tf_simple_relu` is a TensorFlow `Function` that wraps `simple_relu`.
        # tf_simple_relu = tf.function(simple_relu)
        def northvalues(inputimages,yvalue,y):
            # imageset=tf.transpose(inputimages, perm=[0,3,1,2])
            imageset=inputimages
            insize=tf.shape(imageset)
            # tf.print('rollvalues')
            north_loss=[]
            for i in range (insize[0]):
                ix1 = northloss(imageset[i][0])
                ix2 = northloss(imageset[i][1])
                iks = tf_lossim(ix1,ix2)
                # ixs = tf.math.subtract(ix1,ix2)
                # iks = tf.reduce_sum(tf.math.sign(ixs))
                yy=yvalue[i,0]
                iks = tf.reduce_sum(iks)
                # print("north ", iks)
                
                # print("actual north", y[i,0])
                if iks>1:
                    tf.print("north ", iks)
                    tf.print("actual north ",y[i,0])
                    tf.print("predicted", yvalue[i,0])
                    if yy>=0:
                        a=0
                    else:
                        a = abs(yy*iks)
                    # a = [min(yy,0.0)]
                elif iks<-1:
                    tf.print("north ", iks)
                    tf.print("actual north ",y[i,0])
                    tf.print("predicted", yvalue[i,0])
                    if yy<=0:
                        a = 0
                    else:
                        a = abs(yy*iks)
                    # a = [max(yy,0.0)]          
                else:
                    a = 0.0
                north_loss.append(a)
            return tf_sumtensor(tf.abs(north_loss))
        
        def rollvalues(inputimages,yvalue,y):
            # print(yvalue)
            imageset=inputimages
            # imageset=tf.transpose(inputimages, perm=[0,3,1,2])
            insize=tf.shape(inputimages)
            # print(insize)
            roll_loss=[]
            for i in range (insize[0]):
                ix1 = rollloss(imageset[i][0])
                # print(ix1)
                ix2 = rollloss(imageset[i][1])
                iks = tf_lossim(ix1,ix2)
                # ixs = tf.math.subtract(ix1,ix2)
                # iks = tf.cast((tf.reduce_sum(tf.math.sign(ixs))), tf.float32)
                yy=yvalue[i,3]
                # tf.print(yy)
                # print(iks.numpy())
                iks = tf.reduce_sum(iks)
                # print("roll ", iks)
                # print("actual roll ", y[i,3])
                if iks>1:
                    print("roll ", iks)
                    print("actual roll ",y[i,3])
                    if yy>=1:
                        a=0
                    else:
                        a = abs(yy*iks)
                    # a = [min(yy,0.0)]
                elif iks<-1:
                    print("roll ", iks)
                    print("actual roll ",y[i,3])
                    if yy<=-1:
                        a =0
                    else:
                        a = abs(yy*iks)
                    # a = [max(yy,0.0)]          
                else:
                    a = 0.0
                roll_loss.append(a)
            
            # print(roll_loss)
            # rollout=tf.reduce_sum(roll_loss, axis=0)
            # return tf.reduce_sum(roll_loss, axis=0)
            return tf_sumtensor(tf.abs(roll_loss))
        
        @tf.function
        def yawvalues(inputimages,yvalue,y):
            # imageset=tf.transpose(inputimages, perm=[0,3,1,2])
            imageset=inputimages
            insize=tf.shape(imageset)
            # tf.print('rollvalues')
            yaw_loss=[]
            for i in range (insize[0]):
                # for j in range (insize[1]):
                ix1 = yawloss(imageset[i][0])
                ix2 = yawloss(imageset[i][1])
                iks = tf_lossim(ix1,ix2)
                # ixs = tf.math.subtract(ix1,ix2)
                # iks = tf.reduce_sum(tf.math.sign(ixs))
                # print(ix1)
                iks = tf.reduce_sum(iks)
                # print("yaw ",iks)
                # print("actual yaw ",y[i,5])
                yy=yvalue[i,5]
                if iks>0:
                    print("yaw ", iks)
                    print("actual yaw ",y[i,5])
                    if yy>=1:
                        a=0
                    else:
                        a = abs(yy*iks)
                    # a = [min(yy,0.0)]
                elif iks<-1:
                    print("yaw ", iks)
                    print("actual yaw ",y[i,5])
                    if yy<=0:
                        a = 0
                    else:
                        a = abs(yy*iks)
                    # a = [max(yy,0.0)]          
                else:
                    a = 0.0
                yaw_loss.append(a)
            return tf_sumtensor(tf.abs(yaw_loss))
    
        @tf.function
        def pitchvalues(inputimages,yvalue,y):
            # imageset=tf.transpose(inputimages, perm=[0,3,1,2])
            imageset=inputimages
            insize=tf.shape(imageset)
            # tf.print('rollvalues')
            pitch_loss=[]
            for i in range (insize[0]):
                # for j in range (insize[1]):
                ix1 = pitchloss(imageset[i][0])
                ix2 = pitchloss(imageset[i][1])
                iks = tf_lossim(ix1,ix2)
                # ixs = tf.math.subtract(ix1,ix2)
                # iks = tf.reduce_sum(tf.math.sign(ixs))
                yy=yvalue[i,4]
                iks = tf.reduce_sum(iks)
                
                if iks>1:
                    print("pitch ", iks)
                    print("actual pitch ",y[i,4])
                    if yy>=0:
                        a=0
                    else:
                        a = abs(yy*iks)
                    # a = [min(yy,0.0)]
                elif iks<=-1:
                    print("pitch ", iks)
                    print("actual pitch ",y[i,4])
                    if yy<=0:
                        a = 0
                    else:
                        a = abs(yy*iks)
                    # a = [max(yy,0.0)]          
                else:
                    a = 0.0
                pitch_loss.append(a)
            return tf_sumtensor(tf.abs(pitch_loss))
        
        @tf.function
        def tf_for_c_binary(img1,insize):
            x1=[]
            x2=[]
            for i in range(insize):
                x1.append(tf_c_binary(img1[i][0]))
                x2.append(tf_c_binary(img1[i][1]))
            return x1,x2
        
        @tf.function
        def train_step(x,y):
            with tf.GradientTape(persistent=True) as tape:
                logits = model(x, training=True) # Logits for this minibatch
                logits = tf.squeeze(tf.transpose(logits))
                x=tf.transpose(x, [0,3,1,2])
                
                insize = tf.shape(x)
                lam=0.5
                img1=tf_for_c_binary(x, insize[0])
                delnorth = northvalues(img1,logits,y)*lam
                delnorth=0
                delroll = rollvalues(img1,logits,y)*lam
                delpitch = pitchvalues(img1,logits,y)*lam
                delyaw = yawvalues(img1,logits,y)*lam
                lossvalue = north(y,logits)
                # lossvalue=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
                l1error = north_met(y,logits)
                # l1error=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
                lam2=0.1
                delnorth=0
                delroll=0
                delpitch=0
                delyaw=0
                # print(delnorth)
                # print(delroll)
                # print(delpitch)
                # print(delyaw)
                # print('loss', lossvalue)
                
                northl=tf.squeeze(delnorth+lossvalue[0]+lam2*l1error[0])
                eastl=tf.squeeze(lossvalue[1]+lam2*l1error[1])
                upl=tf.squeeze(lossvalue[2]+lam2*l1error[2])
                rolll=tf.squeeze(lossvalue[3]+delroll+lam2*l1error[3])
                pitchl=tf.squeeze(lossvalue[4]+delpitch+lam2*l1error[4])
                yawl=tf.squeeze(lossvalue[5]+delyaw+lam2*l1error[5])
                metricss = north_met(y,logits)
                total_loss = tf.stack((northl,eastl,upl,rolll,pitchl,yawl))
            grads = tape.gradient(total_loss, model.trainable_weights)
                # total_loss=north(y,logits)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            del tape
            
            return total_loss, metricss, tf.stack((delnorth,delroll,delpitch,delyaw))
        @tf.function
        def test_step(x, y):
            val_logits = model(x, training=False)
            # val_acc_metric.update_state(y, val_logits)
        for epoch in range(20):
            print("\nStart of epoch %d" % (epoch,))
            
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(zz):
                # print("\nStep %d" % (step,))
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                # with tf.GradientTape(persistent=True) as tape:
                final_loss, metricss, phyloss = train_step(x_batch_train, y_batch_train)
                
                
                # print("\nStep %d" % (step,), "loss", tf.reduce_sum(final_loss))
                # print("\nStep %d" % (step,), "metrics", tf.reduce_sum(metricss))
                if (step%20)==0:
                    for stepi, (x_batch_val, y_batch_val) in enumerate(valds.take(1)):
                        validationout = model(x_batch_val)
                        validationout=tf.squeeze(tf.transpose(validationout))
                        lossvalue=north(y_batch_val,validationout)
                        metricvalue=north_met(y_batch_val,validationout)
                
                
                    # print("\n validation loss", tf.reduce_mean(lossvalue))
                    # print("\n validation metric", tf.reduce_mean(metricvalue))
                    # print("\nStep %d" % (step,), "loss", tf.reduce_sum(final_loss))
                    # print("\nStep %d" % (step,), "metrics", tf.reduce_sum(metricss))
                    testloss.append(tf.reduce_mean(lossvalue))
                    testmae.append(tf.reduce_mean(metricvalue))
                    # if (step%200)==0:
                        # print(model(x_batch_val))
                trainloss.append(final_loss)
                trainmae.append(metricss)
                
                physicsloss.append(phyloss)
        plt.plot(tf.reduce_sum(trainloss,1))
        return final_loss


vit_classifier = create_vit_classifier()
# vit_classifier.load_weights('vtlstmnew4_0126_04_28')
# tryi = create_vit_classifier2()
y=tf.transpose(vit_classifier.predict(zz.take(1)))
print(y)
# vit_classifier.summary()
history = run_experiment(vit_classifier)

tv = vit_classifier.trainable_variables
tshape=[]
for i in range(len(tv)):
    tshape.append(tf.shape(tv[i]))
    
ty=list(zz.take(1))
ty[0][1]
imgs=ty[0][0]
imgs=tf.transpose(imgs, [0,3,1,2])
x1,x2 = tf_for_c_binary(imgs,10)

tt=northloss(x1[7])
tt2 = northloss(x2[7])
plt.imshow(x1[9])
plt.figure()
plt.imshow(x2[9])

import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('vtlstmnew4_1032.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([trainloss, trainmae, physicsloss], f)

# Getting back the objects:
with open('vtlstmnew4_1032.pkl.pkl') as f:  # Python 3: open(..., 'rb')
    trainloss, trainmae, physicsloss = pickle.load(f)

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