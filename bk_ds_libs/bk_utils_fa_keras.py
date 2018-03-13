# importlib.reload(bk_lib)

import numpy as np

from bk_envs.env_general import *

from bk_ds_libs.OLD import path_manager

sys.path.append(os.path.expanduser(r"~\Learning\dataScience\bk_libs"))

# __t Setup IF Constants => _ANS, _PLOT, _TEST, _IPYTHON
_ANS = True
# _PLOT = True
_PLOT = False  # By default, don't make plots
# _TEST = True
_TEST = False  # Easily enable tests running
# _IPYTHON = True
_IPYTHON = False  # Easily add in code that will only be run line by line in IPython

# __t Setup IMPORT Constants => _ANS, _PLOT, _TEST, _IPYTHON
USE_DEEP_LEARNING = True

# __t DEEP LEARNING
if USE_DEEP_LEARNING:
    # from bk_ds_libs import utils_bk_fa as ut # THIS FILE
    import keras
    from keras.preprocessing import image


# __t ---- STANDARD HEADER FINISHED ----

# __t Getting images as data

# noinspection PyPep8Naming
def get_batches_train_valid(PM: path_manager.PathManager,
                            image_augmentation_multiple=1,
                            batch_size=1) -> (image.ImageDataGenerator, image.ImageDataGenerator):
    """
    train_batches, valid_batches = ut.get_batches_train_valid(PM)
    """
    train_batches = get_batches_augmented(PM.train, image_augmentation_multiple, batch_size=batch_size)
    valid_batches = get_batches(PM.valid, shuffle=False, batch_size=batch_size)
    return train_batches, valid_batches


def get_batches_augmented(dirname,
                          image_augmentation_multiple=1,
                          shuffle=True,
                          batch_size=1,
                          target_size=(224, 224),
                          class_mode="categorical"):
    """train_batches = ut.get_batches_augmented(PM.train)"""
    gen = get_default_image_augmentation_generator(image_augmentation_multiple)
    return get_batches(dirname, gen, shuffle, batch_size, target_size, class_mode)


def get_batches(dirname,
                gen=keras.preprocessing.image.ImageDataGenerator(),
                shuffle=True,
                batch_size=1,
                target_size=(224, 224),
                class_mode="categorical"):
    """Set up with sensible defaults for use with VGG16 and a CPU

    valid_batches = ut.get_batches(PM.valid)
    """
    return gen.flow_from_directory(dirname,
                                   shuffle=shuffle,
                                   batch_size=batch_size,
                                   target_size=target_size,
                                   class_mode=class_mode)


def get_data(dirname='C:/Users/Willem/Desktop/git_repos/fastAI/deeplearning1/nbs/data/dogscats/subsample/valid', target_size=(224, 224)):
    if _IPYTHON:
        dirname = 'C:/Users/Willem/Desktop/git_repos/fastAI/deeplearning1/nbs/data/dogscats/subsample/valid'
        target_size = (224, 224)
    b = get_batches(dirname, target_size=target_size, shuffle=False, batch_size=1)
    list_of_images=[]
    list_of_labels=[]
    # noinspection PyUnusedLocal
    for i in range(b.n):
        imgs, labels = next(b)
        list_of_images.append(imgs[0])
        list_of_labels.append(labels[0])
    # list_of_images = [next(b)[0][0] for i in range(b.n)]
    imgs = np.stack(list_of_images)
    labels = np.stack(list_of_labels)
    if _IPYTHON:
        imgs.shape, labels
        next(b)[0]  # Batch of 1 Image
        next(b)[0].shape
        next(b)[0][0].shape  # 1 Image
        next(b)[1]  # Label
        b.n
        list_of_images
        [im.shape for im in list_of_images]
        d.shape  # (28, 224, 224, 3)
    print(f"def get_data: returning imgs, labels: {imgs.shape}, {labels}")
    return imgs, labels


# __t Plotting a numpy array as an image
# noinspection PyUnusedLocal
def plot_images(ims: tp.Iterable[tp.Union[np.ndarray, "PIL.Image", str]],
                titles=None,
                figsize=(16, 8),
                convert_to_uint8=True):
    """ Plot images horizontally

    :param ims: Iterable of 3 types possible:
        np.ndarray => Expect ims to have shape like (4, 256, 256, 3)
        PIL.Image
        str => filenames of images to load as type PIL.Image
    :param titles: Titles for figures
    :param figsize: (16, X) seems to work well for this
    :param convert_to_uint8: Either have float value in the range [0, 1), or leave this as true!
                             Otherwise image colors get messed up!
    :return:

    EXAMPLE USAGE
    path_data = "C:\\Users\\Willem\\Desktop\\AutoBackupFolder\\organization\\Learning\\dataScience\\bk_ds_libs\\train"
    it = get_batches(path_data, target_size=(224, 224), batch_size=4)
    ims, labels = next(it)
    ims = ims.astype(np.uint8)
    titles = labels # __c TRY BOTH
    """
    # path_data = r"C:\Users\Willem\Desktop\AutoBackupFolder\organization\Learning\dataScience\bk_ds_libs\train"
    if _IPYTHON:
        ims, titles = get_data()
        ims.shape, titles
        titles = None
        # titles = ["title1", np.arange(3)]# # __c TRY BOTH
        figsize = (16, 10)
        convert_to_uint8 = True
        plot_images(ims, titles)
    # print("start of plot_images")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=figsize)
    for i, im in enumerate(ims):
        if _IPYTHON:
            i = 0
            i = 1  # __c TRY BOTH
            im = ims[i]
        print(f"i, im.shape: {i, im.shape}")
        sp = f.add_subplot(1, len(ims), i + 1)
        sp.axis("off")
        title = None if titles is None else titles[i]
        plot_image(im, title, convert_to_uint8)


def plot_image(im, title, convert_to_uint8=True):
    if _IPYTHON:
        imgs, labels = get_data()
        imgs.shape, labels
        im, title = imgs[1], labels[1]
        im, title = imgs[0], labels[0]
        convert_to_uint8 = True
        im.shape, im[:2, :2]
    import matplotlib.pyplot as plt
    if isinstance(im, np.ndarray):
        if im.shape[0] == 3:  # if image shape like: (3, 256, 256), change to (256, 256, 3)
            im = np.moveaxis(im, 0, 2)
        if convert_to_uint8:
            im = im.astype(np.uint8)
    elif isinstance(im, str):  # it's a path
        im = image.load_img(im)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
        # plt.imshow(im[:, :, ::-1])
        # plt.imshow(im[:, :, [2, 0, 1]])
        # return f # Returning this was causing a second plot to be shown in Jupyter Notebook!

if _TEST:
    # __c Notice I set up everything here so that I can run parts of the
    # __c function or of the for loop to check if it's working!
    shape = (5, 7)
    i = 0  # For for loop in ipython - like is first
    im_red = np.stack([np.ones(shape), np.zeros(shape), np.zeros(shape)], -1)
    im_g = np.stack([np.zeros(shape), np.ones(shape), np.zeros(shape)], -1)
    im_b = np.stack([np.zeros(shape), np.zeros(shape), np.ones(shape)], -1)
    im_red.shape
    titles = None  # # __c To test if not passed
    titles = np.array([[0, 1], [1, 0], [1, 0]])
    titles
    # im.shape
    ims = np.stack([im_red, im_g, im_b], axis=0)
    # ims = np.stack([im, im*8+8, im*8], axis=0)
    ims
    plot_images(ims, titles=titles)
    ims[1]

# __Q:  DO
if 'QUESTION: Create and test this function - jnb:KEYtbdJ:':
    if "_ANS" == "2017-09-19":
        # noinspection PyUnreachableCode,PyUnusedLocal
        def deliberate_practice_plot_images(ims: np.ndarray, titles: tp.Optional[tp.List[str]] = None, figsize=(16, 5)):
            """

            :param figsize:
            :param ims: np.ndarray of shape like:  (28, 128, 128, 3)
            :param titles:
            :return:
            """
            if _IPYTHON:
                path_data = r"C:\Users\Willem\Desktop\AutoBackupFolder\organization\Learning\dataScience\bk_libs\train"
                it = get_batches(path_data, target_size=(224, 224), batch_size=4)
                ims, labels = next(it)
                ims = ims.astype(np.uint8)
                ims.shape
                titles = None
                # titles = ["title1", np.arange(3)]# # __c TRY BOTH
                titles = labels  # __c TRY BOTH
                titles
                figsize = (16, 10)
            import matplotlib.pyplot as plt
            f = plt.figure(figsize=figsize)
            for i, im in enumerate(ims):
                if _IPYTHON:
                    # noinspection PyUnusedLocal
                    i = 0
                    i = 1  # __c TRY BOTH
                    im = ims[i]
                sp = f.add_subplot(1, len(ims), i + 1)
                sp.axis('Off')
                plt.imshow(im)
            return f
            if _IPYTHON:
                deliberate_practice_plot_images(ims, titles)
    if _ANS:
        # Above
        pass


def vgg_ft(num_out):
    from bk_vgg16 import Vgg16
    vgg = Vgg16()
    vgg.ft(num_out)
    return vgg.model


def model_predict_on_batches(model, valid_batches):
    return model.predict_generator(valid_batches, valid_batches.n / valid_batches.batch_size, verbose=1)


def model_fit_to_batches(model, train_batches, valid_batches, epochs=1):
    steps_per_epoch_train = train_batches.n / train_batches.batch_size
    steps_per_epoch_valid = valid_batches.n / valid_batches.batch_size
    model.fit_generator(train_batches,
                        steps_per_epoch=steps_per_epoch_train,
                        epochs=epochs,
                        validation_data=valid_batches,
                        validation_steps=steps_per_epoch_valid)


def get_and_save_model_output_on_batches(model: keras.models.Model,
                                         batches: image.DirectoryIterator,
                                         model_output_to="path/to/output/to.bc",
                                         labels_output_to="optional/path/to/output/onehot/labels/to.bc",
                                         num_batches_to_save: int = None):
    # valid_batches = ut.get_batches(path_data_valid, shuffle=False)
    if num_batches_to_save is None:
        num_batches_to_save = int(batches.n / batches.batch_size)
    if model_output_to == "path/to/output/to.bc":
        print('Must provide a path to output data to, such as: path_data_model + "valid_model_out.bc"')
    output_labels = True
    if labels_output_to == "optional/path/to/output/onehot/labels/to.bc":
        output_labels = False

    model_out, labels_out = [], []
    for i in range(num_batches_to_save):
        print(f"batch {i} of {num_batches_to_save}")
        imgs, labels = next(batches)
        model_out.append(model.predict(imgs, batch_size=8,
                                       verbose=1))  # May want to change batch_size on a powerful GPU!
        labels_out.append(labels)
    model_out = np.concatenate(model_out, axis=0)
    labels_out = np.concatenate(labels_out, axis=0)  # This is automatically converted to one-hot!
    print(f"\nmodel_out.shape, labels_out.shape is [[{model_out.shape, labels_out.shape}]]")

    save_array(model_output_to, model_out)
    print(f"Saved model's output to {model_output_to}")
    if output_labels:
        save_array(labels_output_to, labels_out)
        print(f"Saved labels's in one-hot format to {labels_output_to}")


def get_default_image_augmentation_generator(data_augmentation_multiple=1) -> image.ImageDataGenerator:
    """
    Set data_augmentation_multiple=0 to have no augmentations

    :param data_augmentation_multiple:
    :return:
    """
    if data_augmentation_multiple == 0:
        gen = image.ImageDataGenerator()
    else:
        gen = image.ImageDataGenerator(
            rotation_range=15 * data_augmentation_multiple,
            width_shift_range=0.1 * data_augmentation_multiple,
            height_shift_range=0.1 * data_augmentation_multiple,
            zoom_range=0.1 * data_augmentation_multiple,
            horizontal_flip=True)
    return gen


# noinspection PyPep8Naming
def get_and_save_model_output_convFcn(model: keras.models.Model,
                                      path_data: str = "data/dogscats/sample/",
                                      train_or_valid: tp.Union["train", "valid"] = None,
                                      model_description='vgg_conv_GAP',
                                      data_augmentation_multiple=0,
                                      num_epochs=1,
                                      print_helpful_stuff=True):
    """
    EXAMPLE USAGE:
    from bk_ds_libs import bk_vgg_GAP as vgp
    # importlib.reload(vgp)
    m_vgg_conv_GAP = vgp.VGG_Conv_GAP().model
    ut.get_and_save_model_output_convFcn(m_vgg_conv_GAP, PM.data, 'train', model_description='vgg_conv_GAP')
    ut.get_and_save_model_output_convFcn(m_vgg_conv_GAP, PM.data, 'valid', model_description='vgg_conv_GAP')
    ut.get_and_save_model_output_convFcn(m_vgg_conv_GAP, PM.data, 'train', model_description='vgg_conv_GAP',
                                         use_data_augmentation=True, num_epochs=10)
    ut.get_and_save_model_output_convFcn(m_vgg_conv_GAP, PM.data, 'valid', model_description='vgg_conv_GAP',
                                         use_data_augmentation=True, num_epochs=10)

    :param path_data:
    :param model:
    :param train_or_valid:
    :param model_description:
    :param data_augmentation_multiple:
    :param num_epochs:
    :param print_helpful_stuff:
    :return:
    """
    # # __c Finish the usage in the doc string!  See if can easily copy from it!
    assert train_or_valid in "train valid".split()
    path_data_train_or_valid = osp.join(path_data, train_or_valid)
    path_data_model = osp.join(path_data, 'models')

    gen = get_default_image_augmentation_generator(data_augmentation_multiple)

    # region Set up Descriptions
    description_str = f"{train_or_valid}_E{num_epochs}A{data_augmentation_multiple}"
    description_model_out = f'{description_str}_x_{model_description}_out'
    description_y_out = f'{description_str}_y'
    # endregion

    # __t Set up parameters to pass to "get_and_save_model_output_on_batches":
    # region parameters to pass to "get_and_save_model_output_on_batches"
    # model
    batches = get_batches(path_data_train_or_valid, gen, shuffle=False, batch_size=1)
    model_output_to = osp.join(path_data_model, description_model_out + '.bc')
    labels_output_to = osp.join(path_data_model, description_y_out + '.bc')
    num_batches_to_save = num_epochs * batches.n
    # endregion

    get_and_save_model_output_on_batches(model, batches, model_output_to, labels_output_to, num_batches_to_save)

    if print_helpful_stuff:
        xy_vars = f"{description_model_out}, {description_y_out}"
        xy_vars_train = re.sub('valid', 'train', xy_vars)
        xy_vars_valid = re.sub('train', 'valid', xy_vars)
        print(f"""
# Use something like this to load the data PC:KEYcGFr:
{description_model_out} = ut.load_array(osp.join(PM.models, '{description_model_out}.bc'))
{description_y_out} = ut.load_array(osp.join(PM.models, '{description_y_out}.bc'))
{description_model_out}.shape, {description_y_out}.shape

from bk_ds_libs import bk_vgg_GAP as vgp # IF USING bk_vgg_GAP
# importlib.reload(vgp)
mvhg = vgp.VGG_Head_GAP().model
mvhg.fit({xy_vars_train}, batch_size=1, epochs=1, validation_data=({xy_vars_valid}))
""")


print("bk_utils_fa.py loaded")
