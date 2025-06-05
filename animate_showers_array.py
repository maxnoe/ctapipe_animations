from ctapipe.core import Tool, traits
from ctapipe.core.tool import run_tool
from ctapipe.visualization import CameraDisplay
from ctapipe.coordinates import TiltedGroundFrame, GroundFrame
from astropy.coordinates import AltAz
from ctapipe.io import EventSource
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from matplotlib.animation import FuncAnimation
from copy import deepcopy
from tqdm import tqdm


class AnimateArrayShowers(Tool):
    output_path = traits.Path(default_value=None, allow_none=True).tag(config=True)
    fps = traits.Float(default_value=25, help="FPS for the final video").tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("t", "tel"): "EventSource.allowed_tels",
        ("o", "output"): "AnimateArrayShowers.output_path",
        ("m", "max-events"): "EventSource.max_events",
        ('f', 'fps'): 'AnimateArrayShowers.fps',
    }

    classes = traits.classes_with_traits(EventSource)

    def setup(self):
        plt.style.use("dark_background")
        plt.rcParams["axes.facecolor"] = "#00004a"
        plt.rcParams["figure.facecolor"] = "#00004a"

        self.source = self.enter_context(EventSource(parent=self, allowed_tels=[1, 2, 3, 4]))
        subarray = self.source.subarray
        self.events = [event for event in self.source]

        obs_id = self.events[0].index.obs_id
        ob = self.source.observation_blocks[obs_id]


        self.n_events = len(self.events)
        self.n_samples = self.events[0].r1.tel[1].waveform.shape[-1]
        self.frames = self.n_events * self.n_samples


        pointing = AltAz(alt=ob.subarray_pointing_lat, az=ob.subarray_pointing_lon)
        tilted_frame = TiltedGroundFrame(pointing_direction=pointing)
        tel_coords_tilted = subarray.tel_coords.transform_to(tilted_frame)
        tel_x = tel_coords_tilted.x.to_value(u.m)
        tel_y = tel_coords_tilted.y.to_value(u.m)

        # get relative extent to normalize coordinates to [0, 1]
        size = 100

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

        ax = fig.add_axes([0, 0, 1, 1])
        ymax = 200
        ax.set_xlim(-ymax * (16/9), ymax * (16 / 9))
        ax.set_ylim(-ymax, ymax)

        ax.set_aspect(1)
        ax.set_axis_off()

        impact, = ax.plot([], [], 'rx', markeredgewidth=2, ms=15, label='impact', zorder=10)

        axs = [
            ax.inset_axes([x - size / 2, y - size / 2, size, size], transform=ax.transData)
            for x, y in zip(tel_x, tel_y)
        ]

        disps = []
        for i, cam_ax in enumerate(axs):
            geom = deepcopy(subarray.tel[i + 1].camera.geometry)

            disps.append(CameraDisplay(geom, ax=cam_ax, cmap='inferno', show_frame=False))

            # remove any labels / axis / ticks
            cam_ax.set_axis_off()
            cam_ax.set_xticks([])
            cam_ax.set_title('')
            cam_ax.set_xlabel('')
            cam_ax.set_ylabel('')

        self.fig = fig
        self.ax = ax
        self.impact = impact
        self.disps = disps
        self.axs = axs
        self.tilted_frame = tilted_frame
        print("Done setup")

    def start(self):
        with tqdm(total=self.frames + 1) as self.progress:
            ani = FuncAnimation(fig=self.fig, func=self.update, frames=self.frames)

            if self.output_path is not None:
                ani.save(self.output_path, fps=self.fps, dpi=100, savefig_kwargs=dict(facecolor='#00004a'))
            else:
                plt.show()

    def update(self, frame):
        event = self.events[frame // self.n_samples]
        obs_id = event.index.obs_id
        event_id = event.index.event_id
        energy = event.simulation.shower.energy.to_value(u.TeV)
        sample = frame % self.n_samples

        shower = event.simulation.shower
        self.fig.suptitle(f"obs_id: {obs_id}, event_id: {event_id}, energy: {energy:.3f} TeV")

        impact = GroundFrame(x=shower.core_x, y=shower.core_y, z=0 * u.m).transform_to(self.tilted_frame)
        self.impact.set_data(impact.x.to_value(u.m)[..., np.newaxis], impact.y.to_value(u.m)[..., np.newaxis])
        for tel_id, r1 in event.r1.tel.items():
            self.disps[tel_id - 1].image = r1.waveform[0, :, sample]
            self.disps[tel_id - 1].set_limits_minmax(0, r1.waveform[0].max())

        self.progress.update(1)


if __name__ == "__main__":
    AnimateArrayShowers().run()
