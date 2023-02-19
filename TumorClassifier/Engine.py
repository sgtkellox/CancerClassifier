import SlideSegmenter 
import slideio

path = r"D:\Ramin_SS_Oligo_Astro\A2-N17-1152K.svs"

slide = slideio.open_slide(path,'SVS')
num_scenes = slide.num_scenes
scene = slide.get_scene(0)

print(scene.rect[2])
print(scene.rect[3])

segmenter = SlideSegmenter.SlideSegmenter()

segmenter.makeSlicesWithOverlap(1000, 100, scene)
