from train_1705108 import *


def test(X_test_path, model_path, image_size=(28, 28), start_idx=0, samples_to_load=None):
    """
    This function loads the test data and generates the predictions
    :param X_test_path: Path to the test data
    :param model_path: Path to the saved model
    :param image_size: Size of the image
    :param start_idx: Start index of the test data
    :param samples_to_load: Number of samples to load
    """

    filenames = sorted(os.listdir(X_test_path))
    num_samples = len(filenames)

    if start_idx == 0 and samples_to_load is None:
        print("Loading all the test samples")
        samples_to_load = num_samples

    if start_idx + samples_to_load > num_samples:
        print("Invalid start index and number of samples to load")
        return
    
    filenames = filenames[start_idx : start_idx + samples_to_load]
    print("Number of files: " ,len(filenames))

    X_test = []
    for filename in filenames:
        image = Image.open(X_test_path + filename)
        image = np.array(image.resize(image_size))
        X_test.append(image)

    num_channels = 3
    input_image_len = len(np.array(Image.open(X_test_path + filenames[0])).shape)
    if input_image_len == 2:
        num_channels = 1

    X_test = np.array(X_test)
    X_test = (255 - X_test) / 255
    X_test = X_test.reshape(len(X_test), image_size[0], image_size[1], num_channels)

    model_loaded = pickle.load(open(model_path, "rb"))
    y_pred = model_loaded.predict(X_test)

    # y_true = np.argmax(Y_test, axis=1)
    # print("Accuracy: {}" .format(accuracy_score(y_true, y_pred) * 100))

    content = np.vstack((filenames, y_pred)).T
    predictions = pd.DataFrame(content, columns=["image", "Label"])
    # print(y_true)
    # print(y_pred)
    predictions.to_csv("1705108_prediction.csv", index=True)
    print("Predictions generated")


def main(argv):
    X_test_path = argv
    model_path = "1705108_model.pkl"
    test(X_test_path, model_path)

if __name__ == "__main__":
    main(sys.argv[1])