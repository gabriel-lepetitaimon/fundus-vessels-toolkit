from typing import Generator, Tuple

from IPython.display import display
from ipywidgets import GridBox, Layout
from jppype import View2D, imshow, sync_views, vscode_theme
from jppype.layers import Layer, LayerImage, LayerQuiver

vscode_theme()


class Mosaic:
    def __init__(self, rows_cols: int | Tuple[int, int], background=None, cell_height=650, sync=True):
        self.rows = 1 if isinstance(rows_cols, int) else rows_cols[0]
        self.cols = rows_cols if isinstance(rows_cols, int) else rows_cols[1]
        self._cell_height = cell_height

        self._views = [View2D() for _ in range(self.rows * self.cols)]
        for row in range(1, self.rows):
            for col in range(self.cols):
                self[row, col]._top_ruler = not sync
        for col in range(1, self.cols):
            for row in range(self.rows):
                self[row, col]._left_ruler = not sync

        if sync:
            sync_views(*self._views)

        if background is not None:
            self.background = background

    ################################################################################################
    def __len__(self):
        return self.rows * self.cols

    @property
    def views(self) -> Tuple[View2D]:
        return tuple(self._views)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            i, j = index
            index = i * self.cols + j
        try:
            return self._views[index]
        except IndexError:
            raise IndexError("Index out of range") from None

    ################################################################################################
    @property
    def background(self) -> Layer:
        for v in self.views():
            if "background" in v:
                return v["background"]

    @background.setter
    def background(self, value):
        if value is None:
            for v in self.views:
                del v["background"]
        elif isinstance(value, Layer):
            pass
        else:
            value = LayerImage(value)

        value.z_index = -1
        value.foreground = False

        for v in self.views:
            v["background"] = value

    def add_label(self, name: str, label, colormap: str | None = "white", opacity=0.5, options=None, **opts):
        if options is None:
            options = {}
        for v in self.views:
            v.add_label(label, name=name, colormap=colormap, options=options | {"opacity": opacity}, **opts)

    @property
    def cell_height(self):
        return self._cell_height

    @cell_height.setter
    def cell_height(self, value):
        self._cell_height = value

    ################################################################################################
    def _ipython_display_(self):
        self.show()

    def show(self):
        return GridBox(
            self._views,
            layout=Layout(
                grid_template_columns=f"repeat({self.cols}, 1fr)",
                grid_template_rows=f"repeat({self.rows}, {self._cell_height}px)",
            ),
        )
