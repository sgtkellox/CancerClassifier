import SlideSegmenter 
import slideio

path = r"C:\Users\felix\Desktop\eindri\28C9EA57-319D-481A-A722-D46F57109148.svs"

slide = slideio.open_slide(path,'SVS')
num_scenes = slide.num_scenes
scene = slide.get_scene(0)

print(scene.rect[2])
print(scene.rect[3])

segmenter = SlideSegmenter.SlideSegmenter()

segmenter.makeSlicesWithOverlapNoFilter(800, 100, scene)
