import argparse
import tensorflow as tf
from tensorflow import keras


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('multiplicando', type=float, help='Número multiplicando')
    parser.add_argument('multiplicador', type=float, help='Número multiplicador')
    args = parser.parse_args()
    res = multiply(args.multiplicando, args.multiplicador)
    print(f'El resultado de la multiplicación es: {res}')

# Build and train
model_custom_loss   = AutoEncoder(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, L2_Loss)
history_custom_loss = model_custom_loss.fit(X_train_t[0:60], Y_train_t[0:60], epochs=50, 
                                            validation_split=0.3, batch_size=15)

# save model
model_custom_loss.save('drive/MyDrive/ML_project/saved_models/autoencoder_L2')


if __name__ == '__main__':
    main()