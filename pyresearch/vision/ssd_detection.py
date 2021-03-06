import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def print_detection(best_results_per_input,classes_to_labels, img_list):
    inputs = [utils.prepare_input(img) for img in img_list]
    for image_idx in range(len(best_results_per_input)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        ax.imshow(image)
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    
    
    
    
def get_detection(ssd_model, utils , img_list):
    ssd_model.to('cuda')
    ssd_model.eval()
    inputs = [utils.prepare_input(img) for img in img_list]
    tensor = utils.prepare_tensor(inputs, precision == 'fp16')
    detections_batch = ssd_model(tensor)
    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
    return best_results_per_input
    
if __name__ == "__main__":
    precision = 'fp32'
    img_list = ["/media/bharatforge/Ubuntu_data/vishal/datasets/VOC2012/JPEGImages/2007_000027.jpg",
                "/media/bharatforge/Ubuntu_data/vishal/datasets/photos_/IMG_20191001_162245.jpg",
                "/media/bharatforge/Ubuntu_data/vishal/datasets/ODCT_REC_Video/VID_20191001_162850/frame574.jpg",
                "/media/bharatforge/Ubuntu_data/vishal/datasets/ODCT_REC_Video/VID_20191001_162850/frame256.jpg"]
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
    best_results_per_input = get_detection(ssd_model, utils, img_list)
    #print(best_results_per_input)
    classes_to_labels = utils.get_coco_object_dictionary()
    print_detection(best_results_per_input, classes_to_labels, img_list)