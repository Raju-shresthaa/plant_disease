from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6 import QtWidgets
import sys
import os
from PyQt6.uic import loadUiType
import cv2 as cv
import os
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
# from IPython.display import HTML
from keras.preprocessing.image import ImageDataGenerator

ui, _ = loadUiType('homepage.ui')

BATCH_SIZE = 1  # standard batch size
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 1
class MainApp(QWidget, ui):
    def __init__(self):
        print("MainApp: init")

        QWidget.__init__(self)
        self.setupUi(self)
        # self.first_screen()
        self.pushButton.clicked.connect(self.open_camera_tab)
        self.pushButton_4.clicked.connect(self.open_gallery_tab)
        self.pushButton_3.clicked.connect(self.open_report_tab)
        self.pushButton_2.clicked.connect(self.open_exit_tab)
        self.pushButton_8.clicked.connect(self.open_file_explorer)
        self.pushButton_5.clicked.connect(self.load_image)
        self.pushButton_7.clicked.connect(self.backend)

    def backend(self):
        print("delete")
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "Potato",

            seed=123,
            shuffle=True,
            image_size=(256, 256),
            batch_size=32
        )
        class_names = dataset.class_names

        for image_batch, labels_batch in dataset.take(1):
            print(image_batch.shape)
            print(labels_batch.numpy())

        plt.figure(figsize=(15, 15))
        for image_batch, labels_batch in dataset.take(1):
            for i in range(12):
                ax = plt.subplot(3, 4, i + 1)
                plt.imshow(image_batch[i].numpy().astype("uint8"))
                plt.title(class_names[labels_batch[i]])
                plt.axis("off")
        train_size = 0.8
        train_ds = dataset.take(54)
        test_ds = dataset.skip(54)
        val_size = 0.1
        len(dataset) * val_size
        val_ds = test_ds.take(6)
        len(val_ds)
        test_ds = test_ds.skip(6)
        len(test_ds)

        def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                                      shuffle_size=10000):
            assert (train_split + test_split + val_split) == 1

            ds_size = len(ds)

            if shuffle:
                ds = ds.shuffle(shuffle_size, seed=12)

            train_size = int(train_split * ds_size)
            val_size = int(val_split * ds_size)

            train_ds = ds.take(train_size)
            val_ds = ds.skip(train_size).take(val_size)
            test_ds = ds.skip(train_size).skip(val_size)

            return train_ds, val_ds, test_ds

        resize_and_rescale = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(256, 256),
            layers.experimental.preprocessing.Rescaling(1. / 255),
        ])

        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
        ])
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

        aug = ImageDataGenerator(
            rotation_range=25, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2,
            zoom_range=0.2, horizontal_flip=True,
            fill_mode="nearest")

        input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        n_classes = 3

        model = models.Sequential([
            resize_and_rescale,
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(112, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(112, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(112, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(112, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(112, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(112, activation='relu'),
            layers.Dense(n_classes, activation='softmax'),
        ])

        model.build(input_shape=input_shape)

        model.summary()

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        history = model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            validation_data=val_ds,
            verbose=1,
            epochs=1,
        )

        # sc = model.evaluate(test_ds)

        # run prediction
        import numpy as np

        for images_batch, labels_batch in test_ds.take(1):
            first_image = images_batch[0].numpy().astype('uint8')
            first_label = labels_batch[0].numpy()

            print("first image to predict")
            plt.imshow(first_image)
            print("actual label:", class_names[first_label])

            batch_prediction = model.predict(images_batch)
            print("predicted label:", class_names[np.argmax(batch_prediction[0])])

        def predict(model, img):
            img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)

            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = round(100 * (np.max(predictions[0])), 2)
            return predicted_class, confidence

        plt.figure(figsize=(15, 15))
        for images, labels in test_ds.take(1):
            for i in range(12):
                ax = plt.subplot(3, 4, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))

                predicted_class, confidence = predict(model, images[i].numpy())
                actual_class = class_names[labels[i]]

                plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")

                plt.axis("off")
        #

    def load_image(self):
        cap = cv.VideoCapture(0)

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 30
        frame_size = (720, 500)
        video_writer = cv.VideoWriter("D:\vs", fourcc, fps, frame_size)

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                self.first_screen(frame)
                cv.waitKey(0) & 0xFF == ord('q')
                break
            else:
                print("not working")

            cap.release()
            cv.destroyAllWindows()

        # print("Capture Clicked")
        # cap = cv.VideoCapture(0)
        # if not cap.isOpened():
        #     print("Error Camera Open")
        # else:
        #     ret, frame = cap.read()
        #     # cv.imwrite("abc.jpg", frame)
        # cap.release()
        # cv.destroyAllWindows()

    def first_screen(self, img):
        print("display IMage")
        qformat = QImage.Format.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format.Format_RGB888
            else:
                qformat = QImage.Format.Format_RGB888

        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        print("hello")

    def open_camera_tab(self):
        print("Camera Tab")
        self.tabWidget.setCurrentIndex(0)

    def open_gallery_tab(self):
        print("gallery Tab")
        self.tabWidget.setCurrentIndex(1)

    def open_report_tab(self):
        print("Report Tab")
        self.tabWidget.setCurrentIndex(2)

    def open_exit_tab(self):
        print("Exit Tab")
        self.tabWidget.setCurrentIndex(3)

    def open_file_explorer(self):
        print("upload button clicked")
        home_directory = os.path.expanduser("~")
        os.startfile(home_directory)


app = QApplication(sys.argv)
window = MainApp()
window.show()
app.exec()
