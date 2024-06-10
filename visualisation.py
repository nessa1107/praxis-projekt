import matplotlib.pyplot as plt


def visualize_prediction(image, prediction, label):

    image = image.cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(label)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(prediction)
    axes[2].set_title('Model Prediction')
    axes[2].axis('off')
    plt.show()
