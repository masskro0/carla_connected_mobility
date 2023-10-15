import os

import carla
import numpy as np
import pygame

from yolov5.yolo_interface import detect, load_model


class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        absolute_x = pygame.display.Info().current_w - window_size[0]
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (absolute_x, 0)
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def get_display_offset(self, grid_pos):
        dis_size = self.get_display_size()
        return [int(grid_pos[1] * dis_size[0]), int(grid_pos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display is not None


class YOLOCamera:
    def __init__(self, world, display_man, transform, attached, display_pos, device, data, weights, half, img_size,
                 camera_width, camera_height, conf_thres, iou_thres, max_det):
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
        self.sensor = self.init_sensor(transform, attached)
        self.display_man.add_sensor(self)
        self.people_list = []

    def init_sensor(self, transform, attached):
        self.model, self.stride, self.names = load_model(self.device, self.data, self.weights, self.half,
                                                         self.img_size)
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.save_rgb_image)
        self.ready = True
        return camera

    def save_rgb_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        annotated, self.people_list = detect(self.model, array, self.img_size, self.conf_thres, self.iou_thres,
                                             self.max_det, self.names)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(annotated.swapaxes(0, 1))

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()
