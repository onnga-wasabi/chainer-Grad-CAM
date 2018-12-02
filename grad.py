from src.utils.visualize import get_heatmap_imagenet
from src.models import ARCHS
from PIL import Image


model = ARCHS['imagenet']()
image = Image.open('images/leopard.jpg')
get_heatmap_imagenet(model, image)
