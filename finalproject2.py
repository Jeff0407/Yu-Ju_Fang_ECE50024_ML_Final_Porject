import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input, Conv2D, Conv2DTranspose, Flatten, Reshape, Embedding, Concatenate, \
    MaxPool2D, BatchNormalization
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.layers import LeakyReLU, ReLU
import keras.optimizers as optim
from tqdm import tqdm

(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.astype('float32')
X_train = np.expand_dims(X_train, -1)
X_train = X_train / 255
fool_lst = []

def Plot_discriminator_accuracy():
    x = [5*i for i in range(len(fool_lst)-10)]
    y = [sum(fool_lst[i:i+10])/1000 for i in range(len(fool_lst)-10)]
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.plot(x, y)
    plt.savefig(f'/home/fang325/Desktop/optimizer/discriminator.jpg')
    with open('fool_lst_self_Adam.txt', 'w') as f:
        for fool in fool_lst:
            f.write(str(fool))
            f.write('\n')



def Save_figure(latent_dim, generator, discriminator, epoch):
    # plot 10 figures for each number(zero to nine)
    latent_vector = np.random.randn(latent_dim * latent_dim)
    latent_vector = latent_vector.reshape(latent_dim, latent_dim)
    labels = np.array([x for _ in range(10) for x in range(10)]).T

    gen_images = generator.predict([latent_vector, labels])
    fool = 0
    for t in discriminator([gen_images, labels]):
        if float(t) > 0.5:
            fool += 1
    fool_lst.append(fool)
    count = 0

    fig, ax = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            ax[i][j].axis('off')
            ax[i][j].imshow(np.squeeze(gen_images[count], axis=-1), cmap='gray')
            count += 1
    fig.savefig(f'/home/fang325/Desktop/optimizer/gen_images{epoch}.jpg')
    plt.close(fig)


def Collect_real_samples(X_train, y_train, half_batch):
    # Collect half_batch number of real samples
    indx = np.random.randint(0, X_train.shape[0], half_batch)
    return X_train[indx], y_train[indx], np.ones((half_batch, 1))  # Setting the label to 1 to signify real samples.


def Collect_fake_samples(latent_dim, half_batch, generator):
    # Collect half_batch number of fake samples
    X_fake = np.random.randn(latent_dim * half_batch)
    X_fake = X_fake.reshape(half_batch, latent_dim)
    fake_label = np.random.randint(0, 10, half_batch)
    x_fake_img = generator.predict([X_fake, fake_label])
    y_fake = np.zeros((half_batch, 1))  # Setting the label to 0 to signify fake samples.

    return X_fake, fake_label, x_fake_img, y_fake


def Collect_gan_samples(latent_dim, batch_size):
    X_fake_gan = np.random.randn(latent_dim * batch_size)
    gan_label = np.random.randint(0, 10, batch_size)
    X_fake_gan = X_fake_gan.reshape(batch_size, latent_dim)
    y_gan = np.ones((batch_size, 1))

    return X_fake_gan, gan_label, y_gan


def Model_training(epochs, batch_size, X_train, y_train, latent_dim, generator, discriminator, gan):
    batch_per_epoch = X_train.shape[0] // batch_size
    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        d_loss1 = []
        d_loss2 = []
        gan_loss = []
        for _ in tqdm(range(batch_per_epoch)):
            # Discriminator Training
            X_real, real_label, y_real = Collect_real_samples(X_train, y_train, half_batch)
            X_fake, fake_label, x_fake_img, y_fake = Collect_fake_samples(latent_dim, half_batch, generator)

            # Train the generator with both real and fake samples seperately.
            d_loss_real, _ = discriminator.train_on_batch([X_real, real_label], y_real)
            d_loss1.append(d_loss_real)

            d_loss_fake, _ = discriminator.train_on_batch([x_fake_img, fake_label], y_fake)
            d_loss2.append(d_loss_fake)

            # Generator Training
            X_fake_gan, gan_label, y_gan = Collect_gan_samples(latent_dim, batch_size)
            g_l = gan.train_on_batch([X_fake_gan, gan_label], y_gan)

        # Plot 10 generated images per 5 epochs.
        if epoch % 5 == 0:
            Save_figure(latent_dim, generator, discriminator, epoch)
            gan_loss.append(g_l)

        print(
            f'\nEpoch={epoch} DLReal={sum(d_loss1) / (batch_per_epoch)} DLFake={sum(d_loss2) / (batch_per_epoch)} GAN_loss={sum(gan_loss) / batch_per_epoch}')
        if epoch == epochs - 1:
            Plot_discriminator_accuracy()


def Discriminator(in_shape=(28, 28, 1), n_classes=10):
    input_label = Input(shape=(1,), name='discriminator_label_input_layer')
    input_image = Input(shape=in_shape, name='discriminator_image_input_layer')

    embedding_layer = Embedding(n_classes, 50, name='discriminator_label_embedding_layer')(input_label)
    dense = Dense(in_shape[0] * in_shape[1], name='discriminator_label_dense_layer')(embedding_layer)
    reshape = Reshape((in_shape[0], in_shape[1], 1), name='discriminator_label_reshape_layer')(dense)

    merge = Concatenate(name='discriminator_merge_layer')([input_image, reshape])  # Merge the reshape tensor label and the original image

    output = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=in_shape)(merge)
    output = LeakyReLU(alpha=0.02)(output)
    output = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=in_shape)(output)
    output = LeakyReLU(alpha=0.02)(output)
    output = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=in_shape)(output)
    output = LeakyReLU(alpha=0.2)(output)
    output = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(output)

    output = Flatten(name='discriminator_flatten_layer')(output)
    output = Dropout(0.5)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model([input_image, input_label], output, name='Discriminator')
    optimizer = optim.Adam(lr=5*1e-5, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def Generators(latent_dim, n_classes=10, in_shape=(7, 7, 1)):
    input_label = Input((1,), name='generator_label_input_layer')
    latent_vector = Input(shape=(latent_dim,), name='generator_image_latent_dim_input_layer')

    embedding_layer = Embedding(n_classes, 50, name='generator_label_embedding_layer')(input_label)

    dense_label = Dense(in_shape[0] * in_shape[1], name='generator_label_dense_layer')(embedding_layer)
    reshape = Reshape((in_shape[0], in_shape[1], 1), name='generator_label_reshape_layer')(dense_label)

    output_layer = Dense(128 * in_shape[0] * in_shape[1], name='generator_dense_layer')(latent_vector)
    output_layer = LeakyReLU(alpha=0.2, name='generator_activation_layer')(output_layer)
    output_layer = Reshape((in_shape[0], in_shape[1], 128), name='generator_reshape_layer')(output_layer)

    merge = Concatenate(name='generator_merge_layer')([output_layer, reshape])

    model = Sequential()
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(filters=1, kernel_size=(7, 7), activation='sigmoid', padding='same'))
    model = Model([latent_vector, input_label], model(merge), name='Generator')

    return model



def define_gan(generator_model, discriminator_model):
    discriminator_model.trainable = False

    generator_noise, generator_label = generator_model.input
    generator_output = generator_model.output

    gan_output = discriminator_model([generator_output, generator_label])

    model = Model([generator_noise, generator_label], gan_output)
    optimizer = optim.Adam(lr=5*1e-5, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


def main():
    # Define Hyperparameters

    latent_dim = 100
    batch_size = 256
    epochs = 100

    discriminator = Discriminator()  # Generate discriminator
    generator = Generators(latent_dim)  # Generate generators
    gan = define_gan(generator, discriminator)  # Generate gan

    # Training and Save training result as figure
    Model_training(epochs, batch_size, X_train, y_train, latent_dim, generator, discriminator, gan)


if __name__ == '__main__':
    main()
