import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_q


class DisplayManager:
    def __init__(self, window_size):
        self.window_size = (0, 0)
        self.display = None
        self.handle = None

        self.reset(window_size)

    def reset(self, window_size):
        pygame.init()
        pygame.font.init()
        pygame.fastevent.init()

        self.window_size = window_size  # width, height
        self.display = pygame.display.set_mode(window_size, flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.handle = self.launch()
        self.handle.send(None)

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def launch(self):
        """
        Pygame event loop. Since pygame does not support well in a sub thread, use coroutine here.
        """
        running = True
        while running:
            yield
            for event in pygame.fastevent.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        running = False
                        break
        self.close()
        yield

    def step(self):
        if self.handle is not None:
            self.handle.send(None)

    def render(self, image):
        if self.display is not None:
            h, w, _ = image.shape
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            pygame.display.flip()

    def close(self):
        self.display = None
        self.handle = None
        pygame.quit()
