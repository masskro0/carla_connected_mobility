import os

import carla
import numpy as np
import pygame

from yolov5.yolo_interface import detect, load_model


class DisplayManager:
    """Display Manager organizes everything we would like to see in a separate window."""

    def __init__(self, window_size):
        """
        :param window_size: in pixels.
        """
        pygame.init()
        pygame.font.init()

        # Place the window to the upper right corner.
        absolute_x = pygame.display.Info().current_w - window_size[0]
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (absolute_x, 0)
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.window_size = window_size
        self.window_list = []
        self.offsets = []

    def calc_offsets(self):
        current_x = 0
        current_y = 0
        max_x = 0
        for window in self.window_list:
            if current_y + window["height"] > self.window_size[1]:
                current_x += max_x
                current_y = 0
                max_x = window["width"]
            self.offsets.append([current_x, current_y])
            current_y += window["height"]
            if window["width"] > max_x:
                max_x = window["width"]

    def add_window(self, window, width, height):
        """Registers a new window.
        :param window: a new window."""
        self.window_list.append({"window": window, "width": width, "height": height})

    def render(self):
        """Display all frames/windows."""
        if not self.render_enabled():
            return

        for s in self.window_list:
            s["window"].render()

        pygame.display.flip()

    def destroy(self):
        """Cleanup when the code terminates."""
        for s in self.window_list:
            s["window"].destroy()

    def render_enabled(self):
        """Does the display manager renders windows?"""
        return self.display is not None


class PedestrianCamera:
    """Carla camera combined with a YOLO model for object detection."""

    def __init__(self, world, display_man, transform, attached, display_pos, camera_width, camera_height):
        """
        :param world: Carla world instance.
        :param display_man: Display manager instance.
        :param transform: Transformation vector of the camera.
        :param attached: Actor which this camera is attached to.
        :param display_pos: Position inside the display manager.
        :param img_size: Image size.
        :param camera_width: Image width of the Carla camera.
        :param camera_height: Image height of the Carla camera.
        """
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera = self.init_camera(transform, attached)
        self.display_man.add_window(self, self.camera_width, self.camera_height)

    def init_camera(self, transform, attached):
        """Initializes the Carla camera, attaches it to the passed Carla actor, initializes the YOLO model.
        :param transform: Transformation vector of the camera.
        :param attached: Actor which this camera is attached to.
        :return: Carla camera instance.
        """
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.callback)
        return camera

    def callback(self, image):
        """Camera callback function: performs a detection on the new frame.
        :param image: Carla camera image.
        """
        # Convert from Carla raw camera image to numpy array.
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # Create the surface to display the annotated image.
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self):
        """Display the annotated image using the display manager."""
        if self.surface is not None:
            offset = self.display_man.offsets[self.display_pos]
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        """Cleanup when the code gets terminated."""
        self.camera.destroy()


class YOLOCamera:
    """Carla camera combined with a YOLO model for object detection."""

    def __init__(self, world, display_man, transform, attached, display_pos, device, data, weights, half, img_size,
                 camera_width, camera_height, conf_thres, iou_thres, max_det):
        """
        :param world: Carla world instance.
        :param display_man: Display manager instance.
        :param transform: Transformation vector of the camera.
        :param attached: Actor which this camera is attached to.
        :param display_pos: Position inside the display manager.
        :param device: CPU/GPU device name.
        :param data: Path to YOLO data file.
        :param weights: Path to YOLO weights file.
        :param half: Enable 16-bit float precision.
        :param img_size: Image size.
        :param camera_width: Image width of the Carla camera.
        :param camera_height: Image height of the Carla camera.
        :param conf_thres: Confidence threshold at which a detection is accepted.
        :param iou_thres: Intersection-over-Union threshold at which a detection is accepted.
        :param max_det: Maximum number of detections of one image.
        """
        self.names = None
        self.stride = None
        self.model = None
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.device = device
        self.data = data
        self.weights = weights
        self.half = half
        self.img_size = img_size
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.ready = False
        self.camera = self.init_camera(transform, attached)
        self.display_man.add_window(self, self.camera_width, self.camera_height)
        self.people_list = []

    def init_camera(self, transform, attached):
        """Initializes the Carla camera, attaches it to the passed Carla actor, initializes the YOLO model.
        :param transform: Transformation vector of the camera.
        :param attached: Actor which this camera is attached to.
        :return: Carla camera instance.
        """
        self.model, self.stride, self.names = load_model(self.device, self.data, self.weights, self.half,
                                                         self.img_size)
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.detection)
        self.ready = True
        return camera

    def detection(self, image):
        """Camera callback function: performs a detection on the new frame.
        :param image: Carla camera image.
        """
        # Convert from Carla raw camera image to numpy array.
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        annotated, self.people_list = detect(self.model, array, self.img_size, self.conf_thres, self.iou_thres,
                                             self.max_det, self.names)

        # Create the surface to display the annotated image.
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(annotated.swapaxes(0, 1))

    def render(self):
        """Display the annotated image using the display manager."""
        if self.surface is not None:
            offset = self.display_man.offsets[self.display_pos]
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        """Cleanup when the code gets terminated."""
        self.camera.destroy()
