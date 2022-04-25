from gliff import Gliff
import numpy as np
from skimage import img_as_float
from skimage.transform import rescale
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, regionprops_table, label

class Plugin:
    def __init__(self):
        """Initialise the plugin.
        Use the constructor to set up any required attributes needed for the plugin.
        The constructor is run whenever CURATE or ANNOTATE are loaded for a project with this plugin enabled.
        """
        pass

    def __call__(self, image, metadata, annotations):
        """Run morphological geodesic active contours for segmentation.
        Basically nicked from: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_morphsnakes.html
        Use the call method to run your code.
        Inputs:
            image: a PIL Image object
            metadata: a dictionary of metadata
            annotations: a list of annotation objects
        Outputs:
            tuple of updated - 
                image: a PIL Image object
                metadata: a dictionary of metadata
                annotations: a list of annotation objects
        """

        # create an instance of the Gliff class
        gliff = Gliff()

        # Convert image to float
        # NOTE: greyscale images like RBG images have a 3-channels structure
        image = img_as_float(image)[:,:,0]

        thresh = threshold_otsu(image)
        
        mask = image > thresh
        
        # NOTE: for now any metrics can be added to the image metadata using 'gliff.update_metadata_and_labels’ and the matadata can be inspected from CURATE, however in the future we will give you the option to interact with tables from the app.
        particle_measurements = regionprops_table(label(mask),properties=('label','equivalent_daimeter'))# we are also interested in the numbers that we get from this, is there a way to get this csv file out ? 

        level_set = find_contours(mask, level = 0)

        #gliff.add_annotation(annotations, level_set, toolbox="spline") # not sure if the is the correct way to display the mask data - just following the example from below

        # create an annotation from the spline data
        annotations = []
        for indexes in level_set:
            
            # decimate each spline to make it easier to modify them manually from ANNOTATE
            rescaled_indexes = rescale(indexes, 0.05, anti_aliasing=True, channel_axis=1)
            
            # store the indexes as x,y points
            coordinates = [gliff.create_xypoint(x,y) for y,x in rescaled_indexes]

            # create a closed spline
            spline = gliff.create_spline(coordinates, is_closed=True)
            
            # create a spline annotation and add it to the list of annotations
            annotations.append(gliff.create_annotation(toolbox="spline", spline=spline))

        # # Calculate gradient image
        # gradient_image = inverse_gaussian_gradient(image)

        # # Initial level set
        # initial_level_set = np.zeros(image.shape, dtype=np.int8)
        # initial_level_set[10:-10, 10:-10] = 1

        # # List with intermediate results for plotting the evolution
        # level_set = morphological_geodesic_active_contour(gradient_image,
        #                                                   num_iter=230,
        #                                         init_level_set=initial_level_set,
        #                                         smoothing=1, balloon=-1,
        #                                         threshold=0.69)

        # # add level set to annotations as spline
        # gliff.add_annotation(annotations, level_set, toolbox="spline")

        return image, metadata, annotations
