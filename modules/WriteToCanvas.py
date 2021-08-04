from PIL import Image
import numpy as np

def write2canvas(space_dict,images):
  w, h = 4000, 8000
  data = np.zeros((h, w, 3), dtype=np.uint8)
  data[0:8000, 0:4000] = [255,255, 255] 
  canvas = Image.fromarray(data, 'RGB')
  canvas.save('my.png')
  canvas.show()

  offset_w, offset_h = 20,20
  offset = 0, 0
  count=0
  # st = Image.open('/content/words/a01/a01-000u/a01-000u-00-01.png', 'r')
  img = iter(images)
  while(offset_h < 8000):
    while(offset_w < 4000):
      
      #st = Image.open('/content/words/a01/a01-000u/a01-000u-00-0'+str(i)+'.png', 'r')
      if count in space_dict:
        space=space_dict[count]
        if space == 'n':
          break
        if space == 't':
          offset_w = offset_w + 40

      st=next(img)
      st_w, st_h = st.size
      offset = offset_w, offset_h
      canvas.paste(st, offset)
      offset_w =offset_w + st_w + 10
    offset_h =offset_h + 55
    offset_w = 20

  canvas.save('my.png')
