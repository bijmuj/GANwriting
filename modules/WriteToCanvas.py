from PIL import Image
import numpy as np

w, h = 4000, 8000
data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:8000, 0:4000] = [255,255, 255] 
canvas = Image.fromarray(data, 'RGB')
canvas.save('my.png')
canvas.show()

offset_w, offset_h = 20,20
offset = 0, 0
i=0
# st = Image.open('/content/words/a01/a01-000u/a01-000u-00-01.png', 'r')
while(offset_h < 8000):
  while(offset_w < 4000):
    # if i<10:
    #   st = Image.open('/content/words/a01/a01-000u/a01-000u-00-0'+str(i)+'.png', 'r')
    # else:
    #   st = Image.open('/content/words/a01/a01-000u/a01-000u-00-'+str(i)+'.png', 'r')

    st = Image.open('/content/words/a01/a01-000u/a01-000u-00-0'+str(i)+'.png', 'r') # <----------need to change to use in model
    i=i+1
    i=i % 4
    
    st_w, st_h = st.size
    offset = offset_w, offset_h
    canvas.paste(st, offset)
    offset_w =offset_w + st_w + 10
  offset_h =offset_h + 55
  offset_w = 20

canvas.save('my.png')
