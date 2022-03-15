import ast
from svgutils.compose import SVG


def get_true_width_height(svg):
    if "transform" in svg.root.attrib:
        transform = svg.root.attrib["transform"]
        ind = transform.rfind("scale")
        transform_scale = transform[ind:].strip("scale").strip(" ").replace(" ", ",")
        transform_tup = ast.literal_eval(transform_scale)
    else:
        transform_tup = (1.0, 1.0)
    return svg._width.value * transform_tup[0], svg._height.value * transform_tup[1]


class SmartSVG(SVG):
    @property
    def height(self):
        _, height = get_true_width_height(self)
        return height

    @property
    def width(self):
        width, _ = get_true_width_height(self)
        return width

    def set_width(self, width):
        current_width = self.width
        scaler = width / current_width
        self.scale(scaler)

    def set_height(self, height):
        current_height = self.height
        scaler = height / current_height
        self.scale(scaler)
