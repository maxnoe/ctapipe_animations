import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from copy import deepcopy


from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import TelescopeFrame
from ctapipe.io import EventSource
from ctapipe.core import Tool
from ctapipe.core.traits import Float, Path, Int, classes_with_traits


class ShowerAnimation(Tool):
    tel_id = Int(default_value=1, help="Telescope to show").tag(config=True)
    output = Path(help="Output file", allow_none=True, default_value=None).tag(config=True)
    fps = Float(default_value=10, help="FPS for the final video").tag(config=True)

    aliases = {
        ('i', 'input'): 'EventSource.input_url',
        ('o', 'output'): 'ShowerAnimation.output',
        ('t', 'tel'): 'ShowerAnimation.tel_id',
        ('f', 'fps'): 'ShowerAnimation.fps',
    }
    
    classes = classes_with_traits(EventSource)

    def setup(self):
        plt.style.use("dark_background")
        plt.rcParams["axes.facecolor"] = "0.1"
        plt.rcParams["figure.facecolor"] = "0.1"

        self.source = self.enter_context(EventSource(
            parent=self,
            allowed_tels={self.tel_id, }
        ))
        self.subarray = self.source.subarray
        self.cam = self.subarray.tel[self.tel_id].camera.geometry.transform_to(TelescopeFrame())

        self.events = [deepcopy(e) for e in self.source]
        self.n_samples = self.events[0].r1.tel[self.tel_id].waveform.shape[1]

        self.frames = len(self.events) * self.n_samples

        # figsize / dpi will make FullHD video
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        self.ax = self.fig.add_axes([0.01, 0.01, 0.95, 0.94])
        self.ax.set_axis_off()

        self.disp = CameraDisplay(self.cam, cmap='inferno', autoscale=False, show_frame=False)
        self.disp.add_colorbar()
        self.progress = tqdm(total=self.frames)



    def init_ani(self):
        # first sample of the first event
        waveform = self.events[0].r1.tel[self.tel_id].waveform
        self.disp.image = waveform[:, 0]
        self.disp.set_limits_minmax(0, waveform.max())
        return self.disp.pixels,

    def update_ani(self, frame):
        event_index = frame // self.n_samples
        sample = frame % self.n_samples
        event = self.events[event_index]
        waveform = event.r1.tel[self.tel_id].waveform

        self.disp.image = waveform[:, sample]
        if sample == 0:
            self.disp.set_limits_minmax(0, waveform.max())

        obs_id = event.index.obs_id
        event_id = event.index.event_id
        energy = event.simulation.shower.energy.to_value(u.TeV)
        self.disp.axes.set_title(f"obs_id: {obs_id}, event_id: {event_id}, energy: {energy:.3f} TeV")

        self.progress.update(1)
        return self.disp.pixels, self.disp.axes.get_title()

    def start(self):

        ani = FuncAnimation(
            fig=self.fig,
            func=self.update_ani,
            init_func=self.init_ani,
            frames=self.frames,
            interval=1000 / self.fps,
        )
        if self.output is not None:
            ani.save(self.output, fps=self.fps, dpi=100, savefig_kwargs=dict(facecolor='0.1'))
        else:
            plt.show()

if __name__ == '__main__':
    ShowerAnimation().run()
