import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, LSTM, Dense, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------------
# Set seed để đảm bảo tái lập kết quả
# -----------------------------
tf.random.set_seed(42)
np.random.seed(42)
basepath = os.getcwd()

# -----------------------------
# Load dữ liệu train và validation từ file pickle (.p)
# Mỗi file pickle lưu lần lượt input (features) và output (labels)
# -----------------------------
with open(basepath + '/new_data/train_Z24_08_08.p', 'rb') as f:
    input_train = pickle.load(f)   # dữ liệu đầu vào (time series signals)
    output_train = pickle.load(f)  # nhãn tương ứng (damage states)

with open(basepath + '/new_data/valid_Z24_08_08.p', 'rb') as f:
    input_valid = pickle.load(f)
    output_valid = pickle.load(f)    

# -----------------------------
# Đổi shape từ (batch, channels, length) → (batch, length, channels)
# để phù hợp với input của Keras Conv1D/LSTM: (timesteps, features)
# -----------------------------
input_train = input_train.transpose(0,2,1)
input_valid = input_valid.transpose(0,2,1)

# -----------------------------
# Chia dữ liệu train thành 3 phần: train (60%), valid (20%), test (20%)
# Sau đó lại ghép train vào validation → gây data leakage (không chuẩn),
# nhưng đây là cách tác giả code nên giữ nguyên.
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(input_train, output_train, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_valid = np.concatenate((X_valid,X_train))
y_valid = np.concatenate((y_valid,y_train))

# -----------------------------
# In ra danh sách nhãn và số lớp
# -----------------------------
label=np.unique(y_train)
print('Label = ' + str(label))
num_classes = len(np.unique(y_train))
print('No. Labels: ' + str(num_classes))

# -----------------------------
# Hàm xây dựng block ResNet đơn giản cho 1D signal
# -----------------------------
def resnet_block(x, filters, kernel_size=3, stride=1, dilation_rate=1, use_projection_shortcut=False, use_layer_norm=False):
    """ResNet block gồm 3 conv + shortcut"""
    F1, F2, F3 = filters
    shortcut = x
    
    # Conv 1x1
    x = Conv1D(filters=F1, kernel_size=1, strides=stride, dilation_rate=dilation_rate, padding='valid')(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Conv 3x3
    x = Conv1D(filters=F2, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Conv 1x1
    x = Conv1D(filters=F3, kernel_size=1, strides=1, dilation_rate=dilation_rate, padding='valid')(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    
    # Shortcut (nếu cần)
    if use_projection_shortcut:
        shortcut = Conv1D(filters=F3, kernel_size=1, strides=stride, dilation_rate=dilation_rate, padding='valid')(shortcut)
        if use_layer_norm:
            shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# -----------------------------
# Hàm build mô hình 1DCNN + LSTM + ResNet
# -----------------------------
def build_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    
    # Conv đầu vào
    x = Conv1D(filters=64, kernel_size=7, padding="same", strides=2)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2 block ResNet: 1 thường, 1 dùng dilated convolution
    x = resnet_block(x, [64, 64, 256], use_projection_shortcut=True)
    x_dilated = resnet_block(x, [64, 64, 256], use_projection_shortcut=True, dilation_rate=2)
    
    # Merge 2 nhánh
    x = Add()([x, x_dilated])
    
    # LSTM
    # (lưu ý: recurrent_activation='softmax' hơi bất thường, nhưng giữ nguyên code gốc)
    lstm = LSTM(128, return_sequences=True, recurrent_activation='softmax')(x)
    
    # Double skip connections: từ input gốc + từ nhánh ResNet
    shortcut1 = Conv1D(filters=lstm.shape[-1], kernel_size=1, padding="same", strides=2)(input_tensor)
    shortcut1 = BatchNormalization()(shortcut1)
    shortcut2 = Conv1D(filters=lstm.shape[-1], kernel_size=1, padding="same", strides=1)(x)
    shortcut2 = BatchNormalization()(shortcut2)
    
    # Concatenate tất cả
    x = Concatenate(axis=-1)([lstm, shortcut1, shortcut2])
    x = Activation('relu')(x)
    
    # Global pooling
    x_avg = GlobalAveragePooling1D()(x)
    x_max = GlobalMaxPooling1D()(x)
    x = Concatenate(axis=-1)([x_avg, x_max])
    
    # Dense cuối cho phân lớp
    x = Dense(128, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    # Compile model
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# -----------------------------
# Build và train mô hình
# -----------------------------
with tf.device('/GPU:0'):  # chạy trên GPU nếu có
    model_1DCNN_LSTM_ResNet = build_model((5, 8000), num_classes)  # ⚠️ shape này có thể sai trục, nhưng giữ nguyên code gốc
    model_1DCNN_LSTM_ResNet.summary()
    
    # Callback early stopping và checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("model_1DCNN_LSTM_ResNet.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    
    # Train model
    history_1DCNN_LSTM_ResNet = model_1DCNN_LSTM_ResNet.fit(
        X_train, y_train, 
        batch_size=64, epochs=100, 
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping,model_checkpoint]
    )
    
# -----------------------------
# Plot accuracy train/val
# -----------------------------
plt.figure()
plt.plot(history_1DCNN_LSTM_ResNet.history['accuracy'], label='Training')
plt.plot(history_1DCNN_LSTM_ResNet.history['val_accuracy'], label='Validation')
plt.title("Plot Accuracy Training and Validation")
plt.legend()
plt.show()

display(HTML('<hr>'))

# -----------------------------
# Đánh giá trên test set
# -----------------------------
print("--------------test set----------------")
y_pred = model_1DCNN_LSTM_ResNet.predict(X_test,verbose = 0)
y_pred_bool = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_bool)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
report = classification_report(y_test, y_pred_bool,labels=label, output_dict=True)
df_report = pd.DataFrame(report).transpose()
display(df_report.round(3))

display(HTML('<hr>'))

# -----------------------------
# Đánh giá trên validation set
# -----------------------------
print("--------------validate----------------")
y_pred = model_1DCNN_LSTM_ResNet.predict(X_valid,verbose = 0)
y_pred_bool = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_valid, y_pred_bool)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
report = classification_report(y_valid, y_pred_bool,labels=label, output_dict=True)
df_report = pd.DataFrame(report).transpose()
display(df_report.round(3))

display(HTML('<hr>'))

# -----------------------------
# Đánh giá trên train set
# -----------------------------
print("----------------train----------------")
y_pred = model_1DCNN_LSTM_ResNet.predict(X_train,verbose = 0)
y_pred_bool = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_train, y_pred_bool)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
report = classification_report(y_train, y_pred_bool,labels=label, output_dict=True)
df_report = pd.DataFrame(report).transpose()
display(df_report.round(3))

# -----------------------------
# Kết quả cuối cùng: loss & accuracy trên test + validation
# -----------------------------
test_loss_1DCNN_LSTM_ResNet, test_acc_1DCNN_LSTM_ResNet = model_1DCNN_LSTM_ResNet.evaluate(X_test, y_test)
print('Final model has loss of test set is: {} and accuracy is: {}'.format(test_loss_1DCNN_LSTM_ResNet,test_acc_1DCNN_LSTM_ResNet))

val_loss_1DCNN_LSTM_ResNet, val_acc_1DCNN_LSTM_ResNet = model_1DCNN_LSTM_ResNet.evaluate(X_valid, y_valid)
print('Final model has loss of validation set is: {} and accuracy is: {}'.format(val_loss_1DCNN_LSTM_ResNet,val_acc_1DCNN_LSTM_ResNet))
