# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:22:55 2024

@author: PC
"""

#%%
X2 = np.array(X)
power_ch0, phase_ch0 = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,0], SR=250)
power_ch1, phase_ch1 = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,1], SR=250)

X3 = np.array([[power_ch0, power_ch1], [phase_ch0, phase_ch1]]) # fn x ch x n x fi x t


#%%
ingroup = np.logical_or((Z_group[:,6] == 'sham'), (Z_group[:,6] == 'VNS'))
incorrect = (Z_group[:,1] == '맞음')
conditions = [ingroup, incorrect]
vix = np.where(np.logical_and.reduce(conditions))[0]

Xt, Yt = [], []
for i in vix:
    # x1 = np.mean(X3_chmean_fi[i], axis=0)
    X3_tlimit = np.array(X3[:,:,i,:,0:int(SR*(0.25+0.5))])
    X3_axis = X3_tlimit.transpose(0,3,2,1)
    
    rt = float(Z_group[i,2])
    if not(np.isnan(mean_rt)):
        Xt.append(X3_axis)
        Yt.append(rt)
    
Xt, Yt = np.array(Xt), np.array(Yt)
print(Xt.shape, Yt.shape)

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr

# 배열 정규화
Yt = normalize(Yt)
print(Yt)


#%% parallel CNN

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold

# Example data
# Xt = np.random.rand(376, 50, 150, 1)  # Replace with your actual data
# Yt = np.random.rand(376)  # Replace with your actual data

# Define the model architecture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam

def create_parallel_model(input_shape):
    # 첫 번째 입력 및 CNN
    input1 = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation='relu')(input1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(128, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Flatten()(x1)
    
    # 두 번째 입력 및 CNN
    input2 = Input(shape=input_shape)
    x2 = Conv2D(32, (3, 3), activation='relu')(input2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.25)(x2)
    
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.25)(x2)
    
    x2 = Conv2D(128, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.25)(x2)
    
    x2 = Flatten()(x2)
    
    # 병합 및 최종 출력 레이어
    merged = concatenate([x1, x2])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(1)(merged)  # 회귀를 위한 출력 레이어
    
    model = Model(inputs=[input1, input2], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

input_shape = (187, 50, 2)
model = create_parallel_model(input_shape)
model.summary()

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1

spath = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel\\'
for train_index, val_index in kf.split(Xt):
    print(f'Training fold {fold_no} ...')
    
    X_train, X_val = Xt[train_index], Xt[val_index]
    Y_train, Y_val = Yt[train_index], Yt[val_index]
    
    X_train = [X_train[:,0,:,:,:], X_train[:,1,:,:,:]]
    X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
    
    model = create_parallel_model(input_shape)
    
    # Early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(spath + f'model_fold_{fold_no}.h5', save_best_only=True, monitor='val_loss')
    
    # Train the model
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val), 
              callbacks=[early_stopping, model_checkpoint])
    
    fold_no += 1

print('Cross-validation training completed.')



#%% CNN3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold

# Example data
# Xt = np.random.rand(376, 50, 150, 1)  # Replace with your actual data
# Yt = np.random.rand(376)  # Replace with your actual data

# Define the model architecture
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 187, 1)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1))  # Output layer for regression
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1

spath = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel\\'
for train_index, val_index in kf.split(Xt):
    print(f'Training fold {fold_no} ...')
    
    X_train, X_val = Xt[train_index], Xt[val_index]
    Y_train, Y_val = Yt[train_index], Yt[val_index]
    
    model = create_model()
    
    # Early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(spath + f'model_fold_{fold_no}.h5', save_best_only=True, monitor='val_loss')
    
    # Train the model
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val), 
              callbacks=[early_stopping, model_checkpoint])
    
    fold_no += 1

print('Cross-validation training completed.')

#%%

fold_no = 1
predictions = []
ground_truths = []

val_ix_save = []
for train_index, val_index in kf.split(Xt):
    _, X_val = Xt[train_index], Xt[val_index]
    _, Y_val = Yt[train_index], Yt[val_index]
    X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
    
    model = create_parallel_model(input_shape)
    
    # Load the best model for this fold
    best_model = load_model(spath + f'model_fold_{fold_no}.h5')
    # Make predictions on the validation set
    preds = best_model.predict(X_val)
    
    # Store the predictions and ground truths
    predictions.extend(preds)
    ground_truths.extend(Y_val)
    val_ix_save.extend(val_index)
    
    fold_no += 1

# Convert predictions and ground truths to numpy arrays for further analysis
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)
val_ix_save = np.array(val_ix_save)

print('Cross-validation and prediction completed.')

plt.scatter(predictions[:,0], ground_truths, s=5, alpha=0.3)
correlation, p_value = pearsonr(predictions[:,0], ground_truths)
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

#%%

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2

# Function to compute Grad-CAM
def get_gradcam_model(model, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    return grad_model

def compute_gradcam(input_image, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + image
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img


# Load the trained model
fold_no = 1  # Example fold number, adjust as needed
spath = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel\\'
best_model = load_model(spath + f'model_fold_{fold_no}.h5')

# Define the layer to visualize
layer_name = 'conv2d_213'  # Adjust to the appropriate layer name in your model
# layer_name = 'max_pooling2d_92'
# Create Grad-CAM model
gradcam_model = get_gradcam_model(best_model, layer_name)


loss_s = np.square(predictions[:,0] - ground_truths)
mix = np.argsort(loss_s)[:int(len(loss_s)*0.7)]

superimposed_img_save = []
for n in tqdm(mix):
    loss = np.square(predictions[n,0] - ground_truths[n])
    if loss < 0.0005:
        # Example input image
        input_image = Xt[n]  # Replace with your actual input image
        input_image_expanded = np.expand_dims(input_image, axis=0)
        
        # Compute Grad-CAM
        heatmap = compute_gradcam(input_image_expanded, gradcam_model)
        
        # Display Grad-CAM
        input_image_rescaled = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        input_image_rescaled = (input_image_rescaled * 255).astype(np.uint8)
        
        superimposed_img = display_gradcam(heatmap, input_image_rescaled)
        superimposed_img_save.append(superimposed_img/np.sum(superimposed_img))
    
superimposed_img2 = np.mean(np.array(superimposed_img_save), axis=0)
plt.imshow(np.median(superimposed_img2[10:], axis=2))

x_ticks = np.arange(0, superimposed_img2.shape[1], 10)
tix = np.array(np.linspace(0, len(xaxis[s2:e2])-1, len(x_ticks)), dtype=int)
x_labels = np.array(xaxis[s2:e2][tix], dtype=int)

plt.figure(figsize=(10, 10))
plt.imshow(np.mean(superimposed_img2, axis=2), cmap='PiYG')
plt.xticks(ticks=x_ticks, labels=x_labels)

#%%

best_model = load_model(spath + f'model_fold_{fold_no}.h5')
weights, biases = best_model.layers[0].get_weights()
print(weights.shape)  # (3, 3, 1, 32)

fig, axes = plt.subplots(4, 8, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    # 필터 가중치 추출
    img = weights[:, :, 0, i]
    
    # 필터 시각화
    ax.imshow(img, cmap='viridis')
    ax.axis('off')

plt.show()


#%%

stimulus_types = np.array(['neutral', 'congruent', 'incongruent'])
ingroup = np.logical_or((Z_group[:,6] == 'sham'), (Z_group[:,6] == 'VNS'))
incorrect = (Z_group[:,1] == '맞음')

vixsave, vixz = [], []
group_yhat_y = []
mskey_list = list(set(Z_group[:,3]))
for i in range(len(mskey_list)):
    for j in range(len(stimulus_types)):
        for se in range(6):
            stimulus = stimulus_types[j]
            msid = mskey_list[i]
            
            c0 = (Z_group[:,3] == msid)
            c1 = (Z_group[:,0] == stimulus)
            c2 = (Z_group[:,4] == str(se))
            
            conditions = [c0, c1, c2, ingroup, incorrect]
            vix = np.where(np.logical_and.reduce(conditions))[0]
            
            tvix = []
            for k in vix:
                wix = np.where(val_ix_save==k)[0]
                if len(wix) > 0:
                    tvix.append(wix[0])
            if len(tvix) > 0:        
                predictions_group = predictions[tvix,0]
                ground_truths_group = ground_truths[tvix]
                
                nan_indices_pred = np.isnan(predictions_group)
                nan_indices_gt = np.isnan(ground_truths_group)
                nan_indices_combined = nan_indices_pred | nan_indices_gt
                # NaN이 없는 위치의 데이터만 저장
                filtered_predictions = predictions_group[~nan_indices_combined]
                filtered_ground_truths = ground_truths_group[~nan_indices_combined]
                
                # if np.isnan(np.mean(filtered_predictions)):
                #     print(i, j, se)
                        
                group_yhat_y.append([np.mean(filtered_predictions), np.mean(filtered_ground_truths)])
            
group_yhat_y = np.array(group_yhat_y)

plt.scatter(group_yhat_y[:,0], group_yhat_y[:,1])
correlation, p_value = pearsonr(group_yhat_y[:,0], group_yhat_y[:,1])
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")
#%%

import shap
# 모델 로드
fold_no = 1
best_model = load_model(spath + f'model_fold_{fold_no}.h5')

# DeepExplainer 사용
shap_input = [Xt[[0,2,3], 0, :, :, :], Xt[[0,2,3], 1, :, :, :]]
explainer = shap.DeepExplainer(best_model, shap_input)
# SHAP 값 계산
shap_values = explainer.shap_values(shap_input)
# 첫 번째 입력의 SHAP 값 시각화
# 첫 번째 입력의 SHAP 값 시각화
shap_values_input1 = shap_values[0].reshape((shap_values[0].shape[0], -1))
Xt_input1 = Xt[[0,2,3], 0, :, :, :].reshape((Xt[[0,2,3], 0, :, :, :].shape[0], -1))
shap.summary_plot(shap_values_input1, Xt_input1, feature_names=["time:{}_freq:{}".format(i//50, i%50) for i in range(Xt_input1.shape[1])])

# 두 번째 입력의 SHAP 값 시각화
shap_values_input2 = shap_values[1].reshape((shap_values[1].shape[0], -1))
Xt_input2 = Xt[[0,2,3], 1, :, :, :].reshape((Xt[[0,2,3], 1, :, :, :].shape[0], -1))
shap.summary_plot(shap_values_input2, Xt_input2, feature_names=["time:{}_freq:{}".format(i//50, i%50) for i in range(Xt_input2.shape[1])])

            

#%%















